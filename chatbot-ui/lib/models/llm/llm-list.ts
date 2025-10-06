import { LLM } from "@/types"
import { OPENAI_LLM_LIST } from "./openai-llm-list"

export const LLM_LIST: LLM[] = [
  ...OPENAI_LLM_LIST
]

export const LLM_LIST_MAP: Record<string, LLM[]> = {
  openai: OPENAI_LLM_LIST,
  azure: OPENAI_LLM_LIST,
  google: [],
  mistral: [],
  groq: [],
  perplexity: [],
  anthropic: [],
  custom: OPENAI_LLM_LIST,
  openrouter: []
}
