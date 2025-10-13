import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  BaseChatModel,
  BaseChatModelCallOptions,
  BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";
import {
  AIMessageChunk,
  BaseMessage,
  ChatMessage,
  ToolMessage,
  MessageContentComplex,
  DataContentBlock,
  parseBase64DataUrl,
  parseMimeType,
  AIMessage,
  InvalidToolCall,
} from "@langchain/core/messages";
import { ToolCall } from "@langchain/core/messages/tool";
import { ChatResult } from "@langchain/core/outputs";
import { Chat, Query } from "objectiveai";
import { type ClientOptions, OpenAI } from "openai";

// langchain only allows 1 choice, so we must do non-streaming
// so as to put the highest confidence choice first

export interface QueryOptions extends BaseChatModelCallOptions {
  chat_completion_create_params?: Omit<
    Query.Completions.Request.ChatCompletionCreateParamsNonStreaming,
    "messages"
  >;
}

export interface BaseQueryModelParams extends BaseChatModelParams {
  chat_completion_create_params: Omit<
    Query.Completions.Request.ChatCompletionCreateParamsNonStreaming,
    "messages"
  >;
  openai: ClientOptions;
}

export function baseMessageToChatCompletionRequestMessage(
  message: BaseMessage
): Chat.Completions.Request.Message {
  function getMessageTypeOrRole(message: BaseMessage): string {
    const type = message.getType();
    if (type === "generic" && ChatMessage.isInstance(message)) {
      return message.role;
    } else {
      return type;
    }
  }
  function baseMessageContentToUserContent(
    content: string | (MessageContentComplex | DataContentBlock)[]
  ): Chat.Completions.Request.Message.User.Content {
    if (typeof content === "string") {
      return content;
    } else {
      const parts: Chat.Completions.Request.Message.User.Content.Part[] = [];
      for (const p of content) {
        if (p.type === "text" && typeof p.text === "string") {
          parts.push({
            type: "text",
            text: p.text,
          });
        } else if (p.type === "image_url" && typeof p.image_url === "string") {
          parts.push({
            type: "image_url",
            image_url: {
              url: p.image_url,
            },
          });
        } else if (
          p.type === "image_url" &&
          typeof p.image_url === "object" &&
          p.image_url !== null &&
          typeof p.image_url.url === "string"
        ) {
          parts.push({
            type: "image_url",
            image_url: {
              url: p.image_url.url,
              detail:
                p.image_url.detail === "auto" ||
                p.image_url.detail === "low" ||
                p.image_url.detail === "high"
                  ? p.image_url.detail
                  : undefined,
            },
          });
        } else if (
          "source_type" in p &&
          p.source_type === "url" &&
          "url" in p &&
          typeof p.url === "string" &&
          p.type === "image"
        ) {
          parts.push({
            type: "image_url",
            image_url: {
              url: p.url,
            },
          });
        } else if (
          "source_type" in p &&
          p.source_type === "base64" &&
          "data" in p &&
          typeof p.data === "string" &&
          p.type === "image"
        ) {
          parts.push({
            type: "image_url",
            image_url: {
              url: `data:${p.data}`,
            },
          });
        } else if (
          "source_type" in p &&
          p.source_type === "base64" &&
          "data" in p &&
          typeof p.data === "string" &&
          p.type === "audio"
        ) {
          try {
            const parsed = parseBase64DataUrl({
              dataUrl: p.data,
              asTypedArray: false,
            });
            if (parsed) {
              let { subtype } = parseMimeType(parsed.mime_type);
              subtype = subtype.toLowerCase();
              let format: Chat.Completions.Request.Message.User.Content.Part.InputAudio.Format;
              if (subtype === "mpeg" || subtype === "mp3") {
                format = "mp3";
              } else if (subtype === "wav" || subtype === "x-wav") {
                format = "wav";
              } else {
                throw new Error(`Unsupported audio format: ${subtype}`);
              }
              parts.push({
                type: "input_audio",
                input_audio: {
                  data: p.data,
                  format,
                },
              });
            } else {
              throw new Error(
                `Failed to parse audio base64 data URL: ${p.data}`
              );
            }
          } catch (e) {
            throw new Error(
              `Unsupported audio content part: ${p}, error: ${e}`
            );
          }
        } else if (
          "source_type" in p &&
          p.source_type === "base64" &&
          "data" in p &&
          typeof p.data === "string" &&
          p.type === "file"
        ) {
          parts.push({
            type: "file",
            file: {
              file_data: p.data,
            },
          });
        } else {
          throw new Error(`Unsupported Human content part: ${p}`);
        }
      }
      return parts;
    }
  }
  function baseMessageContentToSimpleContent(
    content: string | (MessageContentComplex | DataContentBlock)[]
  ):
    | Chat.Completions.Request.Message.Developer.Content
    | Chat.Completions.Request.Message.System.Content
    | Chat.Completions.Request.Message.Tool.Content {
    if (typeof content === "string") {
      return content;
    } else {
      const parts: (
        | Chat.Completions.Request.Message.Developer.Content.Part
        | Chat.Completions.Request.Message.System.Content.Part
        | Chat.Completions.Request.Message.Tool.Content.Part
      )[] = [];
      for (const p of content) {
        if (p.type === "text" && typeof p.text === "string") {
          parts.push({
            type: "text",
            text: p.text,
          });
        } else {
          throw new Error(`Unsupported Non-Human content part: ${p}`);
        }
      }
      return parts;
    }
  }
  function baseMessageToChatCompletionRequestMessageAssistantToolCalls(
    message: BaseMessage
  ): Chat.Completions.Request.Message.Assistant.ToolCall[] | undefined {
    if ("tool_calls" in message && Array.isArray(message.tool_calls)) {
      const tool_calls: Chat.Completions.Request.Message.Assistant.ToolCall[] =
        [];
      for (const tc of message.tool_calls) {
        if (
          "name" in tc &&
          typeof tc.name === "string" &&
          "args" in tc &&
          typeof tc.args === "object" &&
          tc.args !== null &&
          "id" in tc &&
          typeof tc.id === "string"
        ) {
          try {
            tool_calls.push({
              type: "function",
              id: tc.id,
              function: {
                name: tc.name,
                arguments: JSON.stringify(tc.args),
              },
            });
          } catch (e) {
            throw new Error(
              `Failed to serialize tool call args: ${tc.args}, error: ${e}`
            );
          }
        } else {
          throw new Error(`Unsupported tool call: ${tc}`);
        }
      }
      if (tool_calls.length > 0) {
        return tool_calls;
      }
    }
    return undefined;
  }
  const typeOrRole = getMessageTypeOrRole(message);
  if (typeOrRole === "human" || typeOrRole === "user") {
    return {
      role: "user",
      content: baseMessageContentToUserContent(message.content),
      name: message.name,
    };
  } else if (typeOrRole === "ai" || typeOrRole === "assistant") {
    return {
      role: "assistant",
      content: baseMessageContentToSimpleContent(message.content),
      name: message.name,
      tool_calls:
        baseMessageToChatCompletionRequestMessageAssistantToolCalls(message),
    };
  } else if (typeOrRole === "developer") {
    return {
      role: "developer",
      content: baseMessageContentToSimpleContent(message.content),
      name: message.name,
    };
  } else if (typeOrRole === "system") {
    return {
      role: "system",
      content: baseMessageContentToSimpleContent(message.content),
      name: message.name,
    };
  } else if (typeOrRole === "tool" && ToolMessage.isInstance(message)) {
    return {
      role: "tool",
      content: baseMessageContentToSimpleContent(message.content),
      tool_call_id: message.tool_call_id,
    };
  } else {
    throw new Error(`Unsupported message: ${message}`);
  }
}

export function queryCompletionResponseToBaseMessage(
  completion: Query.Completions.Response.Unary.ChatCompletion
): AIMessage {
  function chatCompletionResponseMessageToBaseMessageToolCalls(
    message: Chat.Completions.Response.Unary.Message
  ): {
    tool_calls?: ToolCall[];
    invalid_tool_calls?: InvalidToolCall[];
  } {
    const tool_calls: ToolCall[] = [];
    const invalid_tool_calls: InvalidToolCall[] = [];
    for (const {
      id,
      function: { name, arguments: args },
    } of message.tool_calls ?? []) {
      try {
        tool_calls.push({
          name,
          args: JSON.parse(args),
          id,
          type: "tool_call",
        });
      } catch (e) {
        invalid_tool_calls.push({
          name,
          args,
          id,
          error: `Failed to parse tool call args: ${args}, error: ${e}`,
          type: "invalid_tool_call",
        });
      }
    }
    return {
      tool_calls: tool_calls.length > 0 ? tool_calls : undefined,
      invalid_tool_calls:
        invalid_tool_calls.length > 0 ? invalid_tool_calls : undefined,
    };
  }
  const choice: Query.Completions.Response.Unary.Choice | undefined =
    completion.choices[0];
  const { tool_calls, invalid_tool_calls } = choice
    ? chatCompletionResponseMessageToBaseMessageToolCalls(choice.message)
    : {};
  return new AIMessage({
    content: JSON.stringify({
      confidence: choice?.confidence ?? 0,
      response: choice?.message.content ?? "",
    }),
    tool_calls,
    invalid_tool_calls,
    id: completion.id,
    response_metadata: {
      id: completion.id,
      choices_count: completion.choices.length,
      created: completion.created,
      model: completion.model,
      object: completion.object,
      service_tier: completion.service_tier,
      system_fingerprint: completion.system_fingerprint,
      choice: {
        finish_reason: choice?.finish_reason,
        index: choice?.index,
        generate_id: choice?.generate_id,
        confidence_id: choice?.confidence_id,
        confidence_weight: choice?.confidence_weight,
        confidence: choice?.confidence,
        model: choice?.model,
        model_index: choice?.model_index,
        completion_metadata: {
          id: choice?.completion_metadata.id,
          created: choice?.completion_metadata.created,
          model: choice?.completion_metadata.model,
          service_tier: choice?.completion_metadata.service_tier,
          system_fingerprint: choice?.completion_metadata.system_fingerprint,
          provider: choice?.completion_metadata.provider,
        },
      },
    },
    usage_metadata: completion.usage
      ? {
          input_tokens: completion.usage.prompt_tokens,
          output_tokens: completion.usage.completion_tokens,
          total_tokens: completion.usage.total_tokens,
          input_token_details: completion.usage.prompt_tokens_details
            ? {
                cache_read:
                  completion.usage.prompt_tokens_details.cached_tokens,
              }
            : undefined,
          output_token_details: completion.usage.completion_tokens_details
            ? {
                reasoning:
                  completion.usage.completion_tokens_details.reasoning_tokens,
              }
            : undefined,
        }
      : undefined,
  });
}

// TODO: support tools
export class QueryObjectiveAI extends BaseChatModel<
  QueryOptions,
  AIMessageChunk
> {
  chat_completion_create_params: Omit<
    Query.Completions.Request.ChatCompletionCreateParamsNonStreaming,
    "messages"
  >;
  openai: ClientOptions;

  constructor(fields: BaseQueryModelParams) {
    super(fields);
    this.chat_completion_create_params = fields.chat_completion_create_params;
    this.openai = fields.openai;
  }

  _llmType() {
    return "objectiveai";
  }

  override invocationParams(_options?: this["ParsedCallOptions"]): {
    chat_completion_create_params: Omit<
      Query.Completions.Request.ChatCompletionCreateParamsNonStreaming,
      "messages"
    >;
    openai: ClientOptions;
  } {
    // use params from constructor as base
    const chat_completion_create_params = {
      ...this.chat_completion_create_params,
    };
    // override with params from call, except for undefined values
    // to delete params from the base, pass in null
    if (_options?.chat_completion_create_params) {
      for (const [key, value] of Object.entries(
        _options.chat_completion_create_params
      )) {
        if (value !== undefined) {
          (chat_completion_create_params as Record<string, unknown>)[key] =
            value;
        }
      }
    }
    // override with disableStreaming
    if (this.disableStreaming) {
      chat_completion_create_params.stream = false;
      chat_completion_create_params.stream_options = undefined;
    }
    return {
      chat_completion_create_params,
      openai: this.openai,
    };
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const {
      chat_completion_create_params: baseChatCompletionCreateParams,
      openai: openaiOptions,
    } = this.invocationParams(options);
    const openai = new OpenAI(openaiOptions);
    const chatCompletionCreateParams: Query.Completions.Request.ChatCompletionCreateParamsNonStreaming =
      {
        ...baseChatCompletionCreateParams,
        messages: messages.map(baseMessageToChatCompletionRequestMessage),
        stream: false,
      };
    const completion = await Query.Completions.create(
      openai,
      chatCompletionCreateParams,
      {
        ...(options?.timeout !== undefined ? { timeout: options.timeout } : {}),
        ...(options?.signal !== undefined ? { signal: options.signal } : {}),
      }
    );
    const baseMessage = queryCompletionResponseToBaseMessage(completion);
    return {
      generations: [
        {
          message: baseMessage,
          text: baseMessage.content as string,
          generationInfo: {
            id: completion.id,
            choices_count: completion.choices.length,
            created: completion.created,
            model: completion.model,
            object: completion.object,
            service_tier: completion.service_tier,
            system_fingerprint: completion.system_fingerprint,
          },
        },
      ],
      llmOutput: {
        tokenUsage: completion.usage
          ? {
              promptTokens: completion.usage.prompt_tokens,
              completionTokens: completion.usage.completion_tokens,
              totalTokens: completion.usage.total_tokens,
              cost: completion.usage.cost,
            }
          : undefined,
      },
    };
  }
}
