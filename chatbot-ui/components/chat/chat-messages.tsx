import { useChatHandler } from "@/components/chat/chat-hooks/use-chat-handler"
import { ChatbotUIContext } from "@/context/context"
import { Tables } from "@/supabase/types"
import { ChatFile } from "@/types"
import { FC, useContext, useState } from "react"
import { Message } from "../messages/message"

interface ChatMessagesProps {}

export const ChatMessages: FC<ChatMessagesProps> = ({}) => {
  const { chatMessages, chatFileItems, newMessageFiles } = useContext(ChatbotUIContext)

  const { handleSendEdit } = useChatHandler()

  const [editingMessage, setEditingMessage] = useState<Tables<"messages">>()

  return chatMessages
    .sort((a, b) => a.message.sequence_number - b.message.sequence_number)
    .map((chatMessage, index, array) => {
      const messageFileItems = chatFileItems.filter(
        (chatFileItem, _, self) =>
          chatMessage.fileItems.includes(chatFileItem.id) &&
          self.findIndex(item => item.id === chatFileItem.id) === _
      )

      // For user messages, also peek at the next assistant message's file items
      let nextMessageFileItems: Tables<"file_items">[] = []
      const next = array[index + 1]
      if (next) {
        nextMessageFileItems = chatFileItems.filter(
          (chatFileItem, _, self) =>
            next.fileItems.includes(chatFileItem.id) &&
            self.findIndex(item => item.id === chatFileItem.id) === _
        )
      }

      // Prefer immediate header files from the compose area for the latest user message
      let headerFiles: ChatFile[] | undefined = undefined
      if (chatMessage.message.role === "user" && index === array.length - 2 && newMessageFiles.length > 0) {
        headerFiles = newMessageFiles
      }

      return (
        <Message
          key={chatMessage.message.sequence_number}
          message={chatMessage.message}
          fileItems={messageFileItems}
          nextMessageFileItems={nextMessageFileItems}
          headerFiles={headerFiles}
          isEditing={editingMessage?.id === chatMessage.message.id}
          isLast={index === array.length - 1}
          onStartEdit={setEditingMessage}
          onCancelEdit={() => setEditingMessage(undefined)}
          onSubmitEdit={handleSendEdit}
        />
      )
    })
}
