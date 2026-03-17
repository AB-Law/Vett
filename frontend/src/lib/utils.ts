import { clsx, type ClassValue } from 'clsx'

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return dateStr
  }
}

export function scoreColor(score: number): string {
  if (score >= 70) return 'score-badge-green'
  if (score >= 50) return 'score-badge-amber'
  return 'score-badge-red'
}

export function scoreColorHex(score: number): string {
  if (score >= 70) return '#4DA044'
  if (score >= 50) return '#F59E0B'
  return '#EF4444'
}
