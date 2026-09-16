defmodule Sagents.Modes.AgentExecutionTest do
  use ExUnit.Case, async: true
  use Mimic

  alias Sagents.Modes.AgentExecution
  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.LangChainError
  alias LangChain.Function
  alias Sagents.MiddlewareEntry
  alias Sagents.Middleware.HumanInTheLoop

  setup :verify_on_exit!

  # ── Helpers ──────────────────────────────────────────────────────

  defp mock_model do
    ChatAnthropic.new!(%{
      model: "claude-sonnet-4-6",
      api_key: "test_key"
    })
  end

  defp build_chain(tools, messages) do
    chain =
      LLMChain.new!(%{
        llm: mock_model(),
        tools: tools
      })

    Enum.reduce(messages, chain, fn msg, acc ->
      LLMChain.add_message(acc, msg)
    end)
  end

  defp submit_tool do
    Function.new!(%{
      name: "submit_report",
      description: "Submit a report",
      parameters_schema: %{
        type: "object",
        properties: %{"title" => %{type: "string"}}
      },
      function: fn args, _ctx -> {:ok, Jason.encode!(args)} end
    })
  end

  defp other_tool do
    Function.new!(%{
      name: "search",
      description: "Search for information",
      parameters_schema: %{
        type: "object",
        properties: %{"query" => %{type: "string"}}
      },
      function: fn args, _ctx -> {:ok, Jason.encode!(args)} end
    })
  end

  defp finalize_tool do
    Function.new!(%{
      name: "finalize",
      description: "Finalize the work",
      parameters_schema: %{
        type: "object",
        properties: %{"status" => %{type: "string"}}
      },
      function: fn args, _ctx -> {:ok, Jason.encode!(args)} end
    })
  end

  # Submit tool whose body validates business rules: only "good" titles
  # succeed; anything else returns {:error, ...} (an error ToolResult).
  defp validating_submit_tool do
    Function.new!(%{
      name: "submit_report",
      description: "Submit a report",
      parameters_schema: %{
        type: "object",
        properties: %{"title" => %{type: "string"}}
      },
      function: fn
        %{"title" => "good"} = args, _ctx -> {:ok, Jason.encode!(args)}
        _args, _ctx -> {:error, "title must be 'good'"}
      end
    })
  end

  defp assistant_with_tool_call(tool_name, args, call_id \\ "call_1") do
    tool_call =
      ToolCall.new!(%{
        status: :complete,
        call_id: call_id,
        name: tool_name,
        arguments: args
      })

    Message.new_assistant!(%{tool_calls: [tool_call]})
  end

  defp plain_assistant_message(content) do
    Message.new_assistant!(%{content: content})
  end

  defp pre_approval_agent(check) do
    test_pid = self()

    tool = %{
      submit_tool()
      | function: fn args, _context ->
          send(test_pid, {:submitted, args})
          {:ok, "submitted"}
        end
    }

    {:ok, agent} =
      Sagents.Agent.new(
        %{
          model: mock_model(),
          tools: [tool],
          tool_context: %{tenant: "tenant-1"},
          middleware: [
            {HumanInTheLoop,
             interrupt_on: %{
               "submit_report" => %{pre_approval: check}
             }}
          ]
        },
        replace_default_middleware: true
      )

    agent
  end

  describe "pre-approval validation" do
    test "rejected batches respect the tool failure retry budget" do
      test_pid = self()

      chain = %{
        build_chain([submit_tool()], [Message.new_user!("Submit a report")])
        | max_retry_count: 1
      }

      {:ok, config} =
        HumanInTheLoop.init(
          interrupt_on: %{
            "submit_report" => %{
              pre_approval: fn _args, _context -> {:error, "Invalid title"} end
            }
          }
        )

      middleware = [%MiddlewareEntry{id: HumanInTheLoop, module: HumanInTheLoop, config: config}]

      stub(ChatAnthropic, :call, fn _model, _messages, _tools ->
        send(test_pid, :model_called)
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "bad"})]}
      end)

      assert {:error, failed_chain, %LangChainError{type: "exceeded_failure_count"}} =
               AgentExecution.run(chain, middleware: middleware, max_runs: 5)

      assert failed_chain.current_failure_count == 1
      assert_received :model_called
      refute_received :model_called
    end

    test "invalid calls return to the model, corrected calls pause, and approved calls execute once" do
      test_pid = self()

      agent =
        pre_approval_agent(fn args, context ->
          send(test_pid, {:checked, context.tenant, context.tool_call_id})
          if args["title"] == "good", do: :ok, else: {:error, "Use a valid title"}
        end)

      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "bad"}, "invalid")]}
      end)
      |> expect(:call, fn _model, messages, _tools ->
        assert %Message{
                 role: :tool,
                 tool_results: [%ToolResult{tool_call_id: "invalid", is_error: true}]
               } = List.last(messages)

        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "good"}, "valid")]}
      end)

      state = Sagents.State.new!(%{messages: [Message.new_user!("Submit the report")]})
      assert {:interrupt, paused, data} = Sagents.Agent.execute(agent, state)
      assert [%{tool_call_id: "valid"}] = data.action_requests
      assert_received {:checked, "tenant-1", "invalid"}
      assert_received {:checked, "tenant-1", "valid"}
      refute_received {:submitted, _}

      stub(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok, [plain_assistant_message("Done")]}
      end)

      assert {:ok, _resumed} = Sagents.Agent.resume(agent, paused, [%{type: :approve}])
      assert_received {:checked, "tenant-1", "valid"}
      assert_received {:submitted, %{"title" => "good"}}
      refute_received {:submitted, _}
    end

    test "a selection that becomes invalid while waiting is rejected on resume" do
      validity = :atomics.new(1, [])

      agent =
        pre_approval_agent(fn _args, _context ->
          if :atomics.get(validity, 1) == 0,
            do: :ok,
            else: {:error, "Selection changed; prepare again"}
        end)

      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "good"})]}
      end)

      state = Sagents.State.new!(%{messages: [Message.new_user!("Submit the report")]})
      assert {:interrupt, paused, _data} = Sagents.Agent.execute(agent, state)
      :atomics.put(validity, 1, 1)

      assert {:ok, resumed} =
               HumanInTheLoop.handle_resume(
                 agent,
                 paused,
                 [%{type: :approve}],
                 hd(agent.middleware).config,
                 []
               )

      assert [%ToolResult{is_error: true} = result] = List.last(resumed.messages).tool_results

      assert LangChain.Message.ContentPart.content_to_string(result.content) =~
               "Selection changed"

      refute_received {:submitted, _}
    end

    test "edited arguments are checked before execution" do
      test_pid = self()

      agent =
        pre_approval_agent(fn args, _context ->
          if args["title"] == "good", do: :ok, else: {:error, "Invalid edited title"}
        end)

      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "good"})]}
      end)

      state = Sagents.State.new!(%{messages: [Message.new_user!("Submit the report")]})
      assert {:interrupt, paused, _data} = Sagents.Agent.execute(agent, state)
      decisions = [%{type: :edit, arguments: %{"title" => "bad"}}]

      callbacks = [
        %{
          on_tool_execution_failed: fn _chain, call, message ->
            send(test_pid, {:failed_edit, call, message})
          end
        }
      ]

      assert {:ok, resumed} =
               HumanInTheLoop.handle_resume(
                 agent,
                 paused,
                 decisions,
                 hd(agent.middleware).config,
                 callbacks: callbacks
               )

      assert [%ToolResult{is_error: true} = result] = List.last(resumed.messages).tool_results

      assert_received {:failed_edit, %ToolCall{call_id: "call_1", arguments: %{"title" => "bad"}},
                       "Invalid edited title"}

      assert LangChain.Message.ContentPart.content_to_string(result.content) =~
               "Invalid edited title"

      refute_received {:submitted, _}
    end

    test "a human denial is not rechecked or turned into retry feedback when a sibling becomes stale" do
      validity = :atomics.new(1, [])

      agent =
        pre_approval_agent(fn _args, context ->
          if :atomics.get(validity, 1) == 0 do
            :ok
          else
            assert context.tool_call_id == "stale"
            {:error, "Stale selection"}
          end
        end)

      expect(ChatAnthropic, :call, fn _model, _messages, _tools ->
        calls =
          for call_id <- ["denied", "stale"] do
            hd(
              assistant_with_tool_call("submit_report", %{"title" => "good"}, call_id).tool_calls
            )
          end

        {:ok, [Message.new_assistant!(%{tool_calls: calls})]}
      end)

      state = Sagents.State.new!(%{messages: [Message.new_user!("Submit the reports")]})
      assert {:interrupt, paused, _data} = Sagents.Agent.execute(agent, state)
      :atomics.put(validity, 1, 1)
      decisions = [%{type: :reject}, %{type: :approve}]

      assert {:ok, resumed} =
               HumanInTheLoop.handle_resume(
                 agent,
                 paused,
                 decisions,
                 hd(agent.middleware).config,
                 []
               )

      assert [denied, stale] = List.last(resumed.messages).tool_results
      assert denied.tool_call_id == "denied"
      assert LangChain.Message.ContentPart.content_to_string(denied.content) =~ "Do not retry"
      assert LangChain.Message.ContentPart.content_to_string(stale.content) =~ "Stale selection"
      refute_received {:submitted, _}
    end
  end

  # ── Test: Standard execution (no until_tool) ─────────────────────

  describe "standard execution (no until_tool)" do
    test "mode runs normally and returns {:ok, chain} when LLM stops" do
      tools = [submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Hello")])

      # First call: LLM returns a tool call
      # Second call: LLM returns a plain assistant message (loop ends)
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "Test"})]}
      end)
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [plain_assistant_message("All done.")]}
      end)

      result = AgentExecution.run(chain, [])

      assert {:ok, %LLMChain{}} = result
    end
  end

  # ── Test: until_tool target tool called ──────────────────────────

  describe "until_tool: target tool called" do
    test "returns {:ok, chain, tool_result} when target tool is called" do
      tools = [submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Write a report")])

      # LLM calls the target tool
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "My Report"})]}
      end)

      result = AgentExecution.run(chain, until_tool: "submit_report")

      assert {:ok, %LLMChain{}, %ToolResult{name: "submit_report"}} = result
    end
  end

  # ── Test: until_tool LLM stops without calling target ────────────

  describe "until_tool: LLM stops without calling target" do
    test "returns error when LLM finishes without calling the target tool" do
      tools = [submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Hello")])

      # LLM returns a plain assistant message (no tool calls)
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [plain_assistant_message("I'm done talking.")]}
      end)

      result = AgentExecution.run(chain, until_tool: "submit_report")

      assert {:error, %LLMChain{}, %LangChainError{type: "until_tool_not_called"} = error} =
               result

      assert error.message =~ "submit_report"
    end
  end

  # ── Test: until_tool with multiple targets ───────────────────────

  describe "until_tool: multiple target tools" do
    test "matches any of the target tools" do
      tools = [submit_tool(), finalize_tool()]
      chain = build_chain(tools, [Message.new_user!("Complete the task")])

      # LLM calls "finalize" which is one of the targets
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("finalize", %{"status" => "complete"})]}
      end)

      result = AgentExecution.run(chain, until_tool: ["submit_report", "finalize"])

      assert {:ok, %LLMChain{}, %ToolResult{name: "finalize"}} = result
    end
  end

  # ── Test: target tool called after multiple iterations ───────────

  describe "until_tool: target called after multiple iterations" do
    test "LLM calls other tools first, then target tool" do
      tools = [other_tool(), submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Research and report")])

      # Iteration 1: LLM calls "search" (not the target)
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("search", %{"query" => "test"}, "call_1")]}
      end)
      # Iteration 2: LLM needs to respond after search results, calls target
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "Found it"}, "call_2")]}
      end)

      result = AgentExecution.run(chain, until_tool: "submit_report")

      assert {:ok, %LLMChain{}, %ToolResult{name: "submit_report"}} = result
    end
  end

  # ── Test: HITL interrupt works with until_tool ───────────────────

  describe "HITL interrupt works with until_tool" do
    test "interrupt short-circuits and returns {:interrupt, chain, data}" do
      tools = [submit_tool()]

      chain =
        build_chain(tools, [Message.new_user!("Write a report")])
        |> LLMChain.update_custom_context(%{
          state: Sagents.State.new!(%{agent_id: "test-hitl-agent"})
        })

      # LLM calls submit_report, but HITL will intercept before tool execution
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "Report"})]}
      end)

      hitl_config = %{
        interrupt_on: %{
          "submit_report" => %{
            allowed_decisions: [:approve, :reject]
          }
        }
      }

      middleware = [
        %MiddlewareEntry{module: HumanInTheLoop, config: hitl_config}
      ]

      result =
        AgentExecution.run(chain,
          until_tool: "submit_report",
          middleware: middleware
        )

      assert {:interrupt, %LLMChain{}, interrupt_data} = result
      assert is_map(interrupt_data)
      assert Map.has_key?(interrupt_data, :action_requests)
    end
  end

  # ── Test: max_runs exceeded with until_tool active ───────────────

  describe "until_tool: max_runs exceeded" do
    test "max_runs exceeded with until_tool active returns error" do
      tools = [other_tool(), submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Research and report")])

      # LLM always calls "search" and never calls "submit_report"
      stub(ChatAnthropic, :call, fn _model, _messages, _tools ->
        tool_call =
          ToolCall.new!(%{
            status: :complete,
            call_id: "call_#{System.unique_integer([:positive])}",
            name: "search",
            arguments: %{"query" => "test"}
          })

        {:ok, [Message.new_assistant!(%{tool_calls: [tool_call]})]}
      end)

      result = AgentExecution.run(chain, until_tool: "submit_report", max_runs: 3)

      assert {:error, %LLMChain{}, %LangChainError{type: "exceeded_max_runs"} = error} = result
      assert error.message =~ "Exceeded maximum number of runs (3/3)"
    end
  end

  describe "require_tool_success: true (retry on error)" do
    test "retries on an error result and terminates on the later success" do
      tools = [validating_submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Write a report")])

      # Iteration 1: bad args -> tool returns {:error, ...} -> loop continues
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "bad"}, "call_1")]}
      end)
      # Iteration 2: corrected args -> success -> terminate
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "good"}, "call_2")]}
      end)

      result =
        AgentExecution.run(chain, until_tool: "submit_report", require_tool_success: true)

      assert {:ok, %LLMChain{}, %ToolResult{name: "submit_report", is_error: false}} = result
    end

    test "exhausts max_runs when the target tool always errors" do
      tools = [validating_submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Write a report")])

      # LLM always submits bad args -> always an error result -> never terminates
      stub(ChatAnthropic, :call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "bad"}, "call_x")]}
      end)

      result =
        AgentExecution.run(chain,
          until_tool: "submit_report",
          require_tool_success: true,
          max_runs: 3
        )

      assert {:error, %LLMChain{}, %LangChainError{type: "exceeded_max_runs"}} = result
    end
  end

  describe "until_tool: target name (stop on call)" do
    test "terminates on the first call even when the result is an error" do
      tools = [validating_submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Write a report")])

      # A single bad call. Call-based until_tool terminates on the error result.
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "bad"}, "call_1")]}
      end)

      result = AgentExecution.run(chain, until_tool: "submit_report")

      assert {:ok, %LLMChain{}, %ToolResult{name: "submit_report", is_error: true}} = result
    end
  end

  # ── Test: normalize_until_tool_opts ──────────────────────────────

  describe "normalize_until_tool_opts (tested through run/2)" do
    test "string is converted to list and active flag" do
      tools = [submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Report")])

      # LLM calls the target tool (verifies normalization happened)
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("submit_report", %{"title" => "Test"})]}
      end)

      result = AgentExecution.run(chain, until_tool: "submit_report")

      assert {:ok, %LLMChain{}, %ToolResult{name: "submit_report"}} = result
    end

    test "list is preserved and active flag is set" do
      tools = [submit_tool(), finalize_tool()]
      chain = build_chain(tools, [Message.new_user!("Finalize")])

      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [assistant_with_tool_call("finalize", %{"status" => "done"})]}
      end)

      result = AgentExecution.run(chain, until_tool: ["submit_report", "finalize"])

      assert {:ok, %LLMChain{}, %ToolResult{name: "finalize"}} = result
    end

    test "nil until_tool means no until_tool behavior" do
      tools = [submit_tool()]
      chain = build_chain(tools, [Message.new_user!("Hello")])

      # LLM returns a plain message (no tool calls), should complete normally
      ChatAnthropic
      |> expect(:call, fn _model, _messages, _tools ->
        {:ok, [plain_assistant_message("Just chatting.")]}
      end)

      result = AgentExecution.run(chain, [])

      assert {:ok, %LLMChain{}} = result
    end
  end
end
