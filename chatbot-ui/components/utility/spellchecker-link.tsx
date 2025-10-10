"use client"

import { Button } from "@/components/ui/button"
import { FC } from "react"

export const SpellcheckerLink: FC = () => {
  const url = process.env.NEXT_PUBLIC_SPELLCHECKER_URL || ""
  if (!url) return null
  return (
    <a href={url} target="_blank" rel="noopener noreferrer">
      <Button variant="outline" size="sm">Spellchecker</Button>
    </a>
  )
}

