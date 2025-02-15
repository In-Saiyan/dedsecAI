import React from 'react'

const About = () => {
    return (
        <div className="flex flex-col justify-center items-center container p-8 gap-6">
            <div className="text-4xl font-monospace font-bold">About Us</div>
            <div className="flex gap-6 flex-col lg:flex-row items-center">
                <div className="scale-90">
                    <h1 className="text-6xl md:text-[6rem] font-black font-monospace">
                        DedSec<span className="text-accent">AI</span>
                    </h1>
                </div>
                <div className="flex flex-col gap-6 text-foreground/80">
                    <p>
                        At DedSec AI, we are transforming post-incident forensic
                        analysis with advanced AI-driven automation. Traditional
                        forensic methods rely on slow, manual processes that are
                        labor-intensive and prone to human error—delaying
                        critical cybersecurity responses.
                    </p>
                    <p>
                        Our platform leverages artificial intelligence and
                        machine learning to streamline forensic investigations,
                        enhancing speed, accuracy, and efficiency. By automating
                        key forensic processes, we empower cybersecurity
                        professionals to detect, analyze, and mitigate threats
                        faster than ever before.
                    </p>
                    <p>
                        With DedSec AI, incident response is no longer
                        reactive—it’s intelligent, proactive, and future-ready.
                    </p>
                </div>
            </div>
        </div>
    )
}

export default About
