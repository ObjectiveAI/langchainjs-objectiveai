import { awaitAllCallbacks } from "@langchain/core/callbacks/promises";
import { afterAll } from "@jest/globals";

afterAll(awaitAllCallbacks);
