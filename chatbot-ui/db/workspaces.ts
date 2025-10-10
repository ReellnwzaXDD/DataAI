import { supabase } from "@/lib/supabase/browser-client"
import { TablesInsert, TablesUpdate } from "@/supabase/types"

export const getHomeWorkspaceByUserId = async (userId: string) => {
  const { data: homeWorkspace, error } = await supabase
    .from("workspaces")
    .select("*")
    .eq("user_id", userId)
    .eq("is_home", true)
    .maybeSingle()

  if (error) {
    throw new Error(error.message)
  }

  if (homeWorkspace) {
    return homeWorkspace.id
  }

  const defaultWorkspace: TablesInsert<"workspaces"> = {
    user_id: userId,
    name: "Home",
    description: "My home workspace.",
    default_context_length: 4096,
    default_model: "dataai",
    default_prompt: "You are a friendly, helpful AI assistant.",
    default_temperature: 0.5,
    include_profile_context: true,
    include_workspace_instructions: true,
    instructions: "",
    is_home: true,
    embeddings_provider: "local",
    sharing: "private"
  }

  const createdWorkspace = await createWorkspace(defaultWorkspace)

  return createdWorkspace.id
}

export const getWorkspaceById = async (workspaceId: string) => {
  const { data: workspace, error } = await supabase
    .from("workspaces")
    .select("*")
    .eq("id", workspaceId)
    .maybeSingle()

  if (error) {
    throw new Error(error.message)
  }

  if (!workspace) {
    throw new Error("Workspace not found")
  }

  return workspace
}

export const getWorkspacesByUserId = async (userId: string) => {
  const { data: workspaces, error } = await supabase
    .from("workspaces")
    .select("*")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })

  if (!workspaces) {
    throw new Error(error.message)
  }

  return workspaces
}

export const createWorkspace = async (
  workspace: TablesInsert<"workspaces">
) => {
  const { data: createdWorkspace, error } = await supabase
    .from("workspaces")
    .insert([workspace])
    .select("*")
    .single()

  if (error) {
    throw new Error(error.message)
  }

  return createdWorkspace
}

export const updateWorkspace = async (
  workspaceId: string,
  workspace: TablesUpdate<"workspaces">
) => {
  const { data: updatedWorkspace, error } = await supabase
    .from("workspaces")
    .update(workspace)
    .eq("id", workspaceId)
    .select("*")
    .single()

  if (error) {
    throw new Error(error.message)
  }

  return updatedWorkspace
}

export const deleteWorkspace = async (workspaceId: string) => {
  const { error } = await supabase
    .from("workspaces")
    .delete()
    .eq("id", workspaceId)

  if (error) {
    throw new Error(error.message)
  }

  return true
}
