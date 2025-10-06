import { LLM } from "@/types"

const DATAAI_PLATFORM_LINK = "http://localhost:3000"

const DATAAI_MODEL: LLM = {
  modelId: "dataai",
  modelName: "DataAi",
  provider: "custom",
  hostedId: "dataai",
  platformLink: DATAAI_PLATFORM_LINK,
  imageInput: false
}

export const OPENAI_LLM_LIST: LLM[] = [DATAAI_MODEL]
