import type { Metadata } from 'next'
import { JetBrains_Mono, Inter } from 'next/font/google'
import { SessionProvider } from 'next-auth/react'
import { auth } from '@/auth'
import AnimatedCursor from 'react-animated-cursor'
import './globals.css'

import ThemeProvider from '@/providers/theme-provider'
import { Navbar } from '@/components/shared/navbar'
import { Footer } from '@/components/shared/footer'
import { Toaster } from '@/components/ui/toaster'

import PageTransition from '@/components/transition/PageTransition'
import StairTransition from '@/components/transition/StairTransition'

const jetbrainsMono = JetBrains_Mono({
    variable: '--font-jetbrainsMono',
    subsets: ['latin'],
    weight: ['100', '200', '300', '400', '500', '600', '700', '800'],
})

const inter = Inter({
    variable: '--font-inter',
    subsets: ['latin'],
})

export const metadata: Metadata = {
    title: 'DedsecAI',
    description: 'A website for the DedsecAI project',
}

export default async function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    const session = await auth()

    return (
        <SessionProvider session={session}>
            <html suppressHydrationWarning lang="en">
                <body
                    className={`${inter.variable} ${jetbrainsMono.variable} antialiased min-h-screen flex flex-col`}
                >
                    <ThemeProvider
                        attribute="class"
                        defaultTheme="system"
                        enableSystem
                        disableTransitionOnChange
                    >
                        <Navbar />
                        <AnimatedCursor
                            innerSize={10}
                            outerSize={30}
                            color="255, 255, 255"
                            outerAlpha={0}
                            innerScale={2}
                            outerScale={3}
                            clickables={[
                                'h1',
                                'h2',
                                'h3',
                                'a',
                                'button',
                                'svg',
                                '.hover-element',
                            ]}
                            outerStyle={{
                                backdropFilter: 'invert(1)',
                                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            }}
                            innerStyle={{
                                backdropFilter: 'invert(1)',
                                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            }}
                        />
                        <StairTransition />
                        <PageTransition>
                            <main className="flex h-full flex-col items-center justify-center">
                                {children}
                            </main>
                        </PageTransition>
                        <Footer />
                        <Toaster />
                    </ThemeProvider>
                </body>
            </html>
        </SessionProvider>
    )
}
