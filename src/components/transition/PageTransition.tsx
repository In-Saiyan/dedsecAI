'use client'
import React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { usePathname } from 'next/navigation'

type childProps = {
    children: React.ReactNode
}

const PageTransition: React.FC<childProps> = ({ children }) => {
    const pathname = usePathname()
    return (
        <AnimatePresence>
            <div key={pathname} className="flex-1">
                <motion.div
                    initial={{ opacity: 1 }}
                    animate={{
                        opacity: 0,
                        transition: {
                            delay: 1,
                            duration: 0.2,
                            ease: 'easeInOut',
                        },
                    }}
                    className="h-screen w-screen fixed dark bg-primary dark:bg-slate-950 top-0 left-0 pointer-events-none z-30"
                    exit={{ opacity: 0 }}
                ></motion.div>
                {children}
            </div>
        </AnimatePresence>
    )
}

export default PageTransition
