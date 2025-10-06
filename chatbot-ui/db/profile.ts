import { supabase } from "@/lib/supabase/browser-client"
import { TablesInsert, TablesUpdate } from "@/supabase/types"

const generateRandomUsername = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `user${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`
  }

  return `user${Math.random().toString(36).replace(/[^a-z0-9]+/gi, "").slice(0, 16)}`
}

export const getProfileByUserId = async (userId: string) => {
  const { data: profile, error } = await supabase
    .from("profiles")
    .select("*")
    .eq("user_id", userId)
    .maybeSingle()

  if (error) {
    throw new Error(error.message)
  }

  if (profile) {
    return profile
  }

  const defaultProfile: TablesInsert<"profiles"> = {
    user_id: userId,
    username: generateRandomUsername(),
    display_name: "",
    bio: "",
    image_url: "",
    image_path: "",
    profile_context: "",
    use_azure_openai: false
  }

  return createProfile(defaultProfile)
}

export const getProfilesByUserId = async (userId: string) => {
  const { data: profiles, error } = await supabase
    .from("profiles")
    .select("*")
    .eq("user_id", userId)

  if (!profiles) {
    throw new Error(error.message)
  }

  return profiles
}

export const createProfile = async (profile: TablesInsert<"profiles">) => {
  const { data: createdProfile, error } = await supabase
    .from("profiles")
    .insert([profile])
    .select("*")
    .single()

  if (error) {
    throw new Error(error.message)
  }

  return createdProfile
}

export const updateProfile = async (
  profileId: string,
  profile: TablesUpdate<"profiles">
) => {
  const { data: updatedProfile, error } = await supabase
    .from("profiles")
    .update(profile)
    .eq("id", profileId)
    .select("*")
    .single()

  if (error) {
    throw new Error(error.message)
  }

  return updatedProfile
}

export const deleteProfile = async (profileId: string) => {
  const { error } = await supabase.from("profiles").delete().eq("id", profileId)

  if (error) {
    throw new Error(error.message)
  }

  return true
}
