import * as React from 'react'

import { cn } from '@/lib/utils'

const Textarea = React.forwardRef<
    HTMLTextAreaElement,
    React.ComponentProps<'textarea'>
>(({ className, ...props }, ref) => {
    return (
        <textarea
            className={cn(
                'flex min-h-[80px] w-full rounded-md py-2 text-base ring-offset-background focus-visible:ring-1 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50 md:text-sm h-[120px] ring-1 ring-primary/20 border-none focus:ring-accent bg-background text-foreground placeholder:text-foreground/60 px-4 outline-none active:outline-none focus:outline-none focus-visible:outline-none focus-visible:ring-accent',
                className
            )}
            ref={ref}
            {...props}
        />
    )
})
Textarea.displayName = 'Textarea'

export { Textarea }
