import { Serializable } from "@langchain/core/load/serializable";
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
  output: T;
}

function throwNotObjectiveAIResponseError(): never {
  throw new Error(
    "ObjectiveAI Query parser should only be used with an ObjectiveAI Query response."
  );
}

function parseObjectiveAIResponse(
  text: string
): QueryObjectiveAIOutput<string>[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throwNotObjectiveAIResponseError();
  }
  if (!Array.isArray(parsed)) {
    throwNotObjectiveAIResponseError();
  }
  const content: QueryObjectiveAIOutput<string>[] = [];
  for (const item of parsed) {
    if (
      typeof item === "object" &&
      item !== null &&
      "output" in item &&
      typeof item.output === "string" &&
      "confidence" in item &&
      typeof item.confidence === "number"
    ) {
      content.push({
        confidence: item.confidence,
        output: item.output,
      });
    } else {
      throwNotObjectiveAIResponseError();
    }
  }
  return content;
}

export class QueryObjectiveAITextOutputParser extends BaseOutputParser<
  QueryObjectiveAIOutput<string>[]
> {
  constructor() {
    super();
  }

  lc_namespace = ["langchain", "output_parsers", "objectiveai"];

  parse(text: string): Promise<QueryObjectiveAIOutput<string>[]> {
    return Promise.resolve(parseObjectiveAIResponse(text));
  }

  getFormatInstructions(): string {
    return "";
  }
}

export class QueryObjectiveAIJsonObjectOutputParser extends BaseOutputParser<
  QueryObjectiveAIOutput<Record<string, unknown>>[]
> {
  structuredParser: StructuredOutputParser<any>;

  constructor() {
    super();
    this.structuredParser = new StructuredOutputParser(
      z.record(z.unknown()) as any
    );
  }

  lc_namespace = ["langchain", "output_parsers", "objectiveai"];

  async parse(
    text: string
  ): Promise<QueryObjectiveAIOutput<Record<string, unknown>>[]> {
    const choicesRaw = parseObjectiveAIResponse(text);
    // only throw for the winning choice, otherwise omit failures
    const choicesPromises = choicesRaw.map(
      async ({ confidence, output: rawOutput }, i) => {
        if (i === 0) {
          const output = (await this.structuredParser.parse(
            rawOutput
          )) as Record<string, unknown>;
          return { confidence, output };
        } else {
          try {
            const output = (await this.structuredParser.parse(
              rawOutput
            )) as Record<string, unknown>;
            return { confidence, output };
          } catch {
            return undefined;
          }
        }
      }
    );
    const choices = await Promise.all(choicesPromises);
    return choices.filter((c) => c !== undefined);
  }

  getFormatInstructions(): string {
    return "";
  }
}

export class QueryObjectiveAIJsonSchemaOutputParser<
  T extends InteropZodType
> extends BaseOutputParser<QueryObjectiveAIOutput<InferInteropZodOutput<T>>[]> {
  structuredParser: StructuredOutputParser<T>;

  constructor(schema: T) {
    super(schema);
    this.structuredParser = new StructuredOutputParser(schema);
  }

  lc_namespace = ["langchain", "output_parsers", "objectiveai"];

  async parse(
    text: string
  ): Promise<QueryObjectiveAIOutput<InferInteropZodOutput<T>>[]> {
    const choicesRaw = parseObjectiveAIResponse(text);
    // only throw for the winning choice, otherwise omit failures
    const choicesPromises = choicesRaw.map(
      async ({ confidence, output: rawOutput }, i) => {
        if (i === 0) {
          const output = await this.structuredParser.parse(rawOutput);
          return { confidence, output };
        } else {
          try {
            const output = await this.structuredParser.parse(rawOutput);
            return { confidence, output };
          } catch {
            return undefined;
          }
        }
      }
    );
    const choices = await Promise.all(choicesPromises);
    return choices.filter((c) => c !== undefined);
  }

  getFormatInstructions(): string {
    return "";
  }
}

export class QueryObjectiveAICustomOutputParser<T> extends BaseOutputParser<
  QueryObjectiveAIOutput<T>[]
> {
  parseFunction: (text: string) => Promise<T>;

  constructor(
    parseFunction: (text: string) => Promise<T>,
    kwargs?: ConstructorParameters<typeof Serializable>[0],
    ..._args: never[]
  ) {
    super(kwargs, ..._args);
    this.parseFunction = parseFunction;
  }

  lc_namespace = ["langchain", "output_parsers", "objectiveai"];

  async parse(text: string): Promise<QueryObjectiveAIOutput<T>[]> {
    const choicesRaw = parseObjectiveAIResponse(text);
    // only throw for the winning choice, otherwise omit failures
    const choicesPromises = choicesRaw.map(
      async ({ confidence, output: rawOutput }, i) => {
        if (i === 0) {
          const output = await this.parseFunction(rawOutput);
          return { confidence, output };
        } else {
          try {
            const output = await this.parseFunction(rawOutput);
            return { confidence, output };
          } catch {
            return undefined;
          }
        }
      }
    );
    const choices = await Promise.all(choicesPromises);
    return choices.filter((c) => c !== undefined);
  }

  getFormatInstructions(): string {
    return "";
  }
}
