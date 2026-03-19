import { Link } from 'react-router-dom'
import { Bot, BookOpen, BrainCircuit, Network } from 'lucide-react'

const modules = [
  {
    id: 'flashcards',
    title: 'Flashcard Decks',
    description: 'Create and review named decks with selected interview documents.',
    to: '/study/flashcards',
    cta: 'Open Decks',
    disabled: false,
    icon: BookOpen,
  },
  {
    id: 'quiz',
    title: 'Quiz Mode',
    description: 'Adaptive timed quizzes are coming soon.',
    to: '/study',
    cta: 'Coming soon',
    disabled: true,
    icon: BrainCircuit,
  },
  {
    id: 'mindmap',
    title: 'Concept Mind Map',
    description: 'Extract and explore a concept graph from interview knowledge documents.',
    to: '/study/mindmap',
    cta: 'Open Mind Map',
    disabled: false,
    icon: Network,
  },
  {
    id: 'interview',
    title: 'Interview Simulator',
    description: 'Jump to live interview simulation flows from your jobs list.',
    to: '/jobs',
    cta: 'Go to Jobs',
    disabled: false,
    icon: Bot,
  },
]

export default function StudyHub() {
  return (
    <div className="study-page">
      <section className="study-header">
        <h1>Study Hub</h1>
        <p>Pick a module to practice interview prep in the format you want.</p>
      </section>

      <section className="study-modules-grid">
        {modules.map((module) => {
          const Icon = module.icon
          return (
            <article key={module.id} className={`study-module-card ${module.disabled ? 'study-module-card-disabled' : ''}`}>
              <div className="study-module-icon-wrap">
                <Icon className="w-5 h-5" />
              </div>
              <div className="space-y-2">
                <h2 className="section-title">{module.title}</h2>
                <p className="text-sm text-text-muted">{module.description}</p>
              </div>
              {module.disabled ? (
                <span className="study-module-cta-disabled">{module.cta}</span>
              ) : (
                <Link className="study-module-cta" to={module.to}>
                  {module.cta}
                </Link>
              )}
            </article>
          )
        })}
      </section>
    </div>
  )
}
