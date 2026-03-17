import { NavLink, Outlet } from 'react-router-dom'
import {
  LayoutDashboard,
  FileText,
  ClipboardCheck,
  Briefcase,
  Settings,
  Wifi,
  TrendingUp,
} from 'lucide-react'
import { cn } from '../lib/utils'

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/cv', icon: FileText, label: 'My CV' },
  { to: '/score', icon: ClipboardCheck, label: 'Score JD' },
  { to: '/jobs', icon: Briefcase, label: 'Jobs' },
]

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-canvas">
      {/* Sidebar */}
      <aside className="w-[220px] bg-sidebar flex flex-col shrink-0 overflow-hidden">
        {/* Logo */}
        <div className="px-4 py-5 flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-sage-600/20 flex items-center justify-center">
            <TrendingUp className="w-4 h-4 text-sage-400" />
          </div>
          <span className="text-white font-semibold text-[15px] tracking-tight">Vett</span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 flex flex-col gap-0.5 mt-2">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2 rounded text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-white/10 text-white'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <Icon
                    className={cn('w-4 h-4 shrink-0', isActive ? 'text-sage-400' : '')}
                  />
                  {label}
                </>
              )}
            </NavLink>
          ))}

          <div className="mt-4 border-t border-white/5 pt-2">
            <NavLink
              to="/settings"
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2 rounded text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-white/10 text-white'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <Settings className={cn('w-4 h-4 shrink-0', isActive ? 'text-sage-400' : '')} />
                  Settings
                </>
              )}
            </NavLink>
          </div>
        </nav>

        {/* Local Mode Badge */}
        <div className="p-3">
          <div className="flex items-center gap-2 px-3 py-2 rounded bg-white/5 border border-white/8">
            <Wifi className="w-3.5 h-3.5 text-sage-400" />
            <span className="text-xs text-gray-400 font-medium">Local Mode</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-canvas">
        <Outlet />
      </main>
    </div>
  )
}
