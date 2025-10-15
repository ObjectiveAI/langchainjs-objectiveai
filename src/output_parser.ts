import {
  BaseOutputParser,
  StructuredOutputParser,
} from "@langchain/core/output_parsers";
import {
  InferInteropZodOutput,
  InteropZodType,
} from "@langchain/core/utils/types";
import { z } from "zod";

export interface QueryObjectiveAIOutput<T> {
  confidence: number;
  response: T;
}

function throwNotObjectiveAIResponseError(): never {
  throw new Error(
    "ObjectiveAI Query parser should only be used with an ObjectiveAI Query response."
  );
}

function parseObjectiveAIResponse(
  text: string
): QueryObjectiveAIOutput<string> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throwNotObjectiveAIResponseError();
  }
  if (
    typeof parsed === "object" &&
    parsed !== null &&
    "response" in parsed &&
    typeof parsed.response === "string" &&
    "confidence" in parsed &&
    typeof parsed.confidence === "number" &&
    parsed.confidence >= 0 &&
    parsed.confidence <= 1
  ) {
    return { confidence: parsed.confidence, response: parsed.response };
  } else {
    throwNotObjectiveAIResponseError();
  }
}

export class QueryObjectiveAITextOutputParser extends BaseOutputParser<
  QueryObjectiveAIOutput<string>
> {
  constructor() {
    super();
  }

  lc_namespace = ["langchain", "output_parsers", "objectiveai"];

  parse(text: string): Promise<QueryObjectiveAIOutput<string>> {
    return Promise.resolve(parseObjectiveAIResponse(text));
  }

  getFormatInstructions(): string {
    return "";
  }
}

export class QueryObjectiveAIJsonObjectOutputParser extends BaseOutputParser<
  QueryObjectiveAIOutput<Record<string, unknown>>
> {
  structuredParser: StructuredOutputParser<
    z.ZodRecord<z.ZodString, z.ZodUnknown>
  >;

  constructor() {
    const schema = z.record(z.string(), z.unknown());
    super(schema);
    this.structuredParser = new StructuredOutputParser(schema);
  }

  lc_namespace = ["langchain", "output_parsers", "objectiveai"];

  async parse(
    text: string
  ): Promise<QueryObjectiveAIOutput<Record<string, unknown>>> {
    const { confidence, response: rawResponse } =
      parseObjectiveAIResponse(text);
    const response = await this.structuredParser.parse(rawResponse);
    return { confidence, response };
  }

  getFormatInstructions(): string {
    return "";
  }
}

export class QueryObjectiveAIJsonSchemaOutputParser<
  T extends InteropZodType
> extends BaseOutputParser<QueryObjectiveAIOutput<InferInteropZodOutput<T>>> {
  structuredParser: StructuredOutputParser<T>;

  constructor(schema: T) {
    super(schema);
    this.structuredParser = new StructuredOutputParser(schema);
  }

  lc_namespace = ["langchain", "output_parsers", "objectiveai"];

  async parse(
    text: string
  ): Promise<QueryObjectiveAIOutput<InferInteropZodOutput<T>>> {
    const { confidence, response: rawResponse } =
      parseObjectiveAIResponse(text);
    const response = await this.structuredParser.parse(rawResponse);
    return { confidence, response };
  }

  getFormatInstructions(): string {
    return "";
  }
}
