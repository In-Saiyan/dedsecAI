import * as React from 'react'

import { cn } from '@/lib/utils'

export interface InputProps
    extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
    ({ className, type, ...props }, ref) => {
        return (
            <input
                type={type}
                className={cn(
                    'flex w-full border border-input py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:cursor-not-allowed disabled:opacity-50 h-[48px] ring-1 focus-visible:ring-1 ring-primary/20 rounded-md border-none focus:border-accent focus:ring-accent bg-background text-foreground placeholder:text-foreground/60 px-4 outline-none active:outline-none focus:outline-none focus-visible:outline-none focus-visible:ring-accent',
                    className
                )}
                ref={ref}
                {...props}
            />
        )
    }
)
Input.displayName = 'Input'

export { Input }
