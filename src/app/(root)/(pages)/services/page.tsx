'use client'
import Link from 'next/link'
import React from 'react'
import { BsArrowDownRight } from 'react-icons/bs'
import { motion } from 'framer-motion'

const services = [
    {
        num: '01',
        title: 'Dashboard with Incident Overview',
        href: '/dashboard',
        description:
            'Real-time visualization of ongoing & past security incidents with an attack timeline and interactive graphs.',
    },
    {
        num: '02',
        title: 'AI-Powered Attack Vector Analysis',
        href: '/attack-analysis',
        description:
            'Automatically classify attack types (e.g., phishing, malware, SQL injection) and provide AI-generated insights on breach origins.',
    },
    {
        num: '03',
        title: 'Log File Analyzer',
        href: '/log-analyzer',
        description:
            'Upload log files for AI to detect suspicious patterns, with support for different log formats like JSON, Syslog, etc.',
    },
    {
        num: '04',
        title: 'Threat Intelligence Integration',
        href: '/threat-intelligence',
        description:
            'Fetch data from sources like VirusTotal, AlienVault, and MITRE ATT&CK, and compare attack patterns with global threat databases.',
    },
    {
        num: '05',
        title: 'Automated Incident Response Recommendations',
        href: '/incident-response',
        description:
            'AI suggests actions to mitigate & contain breaches, with a step-by-step remediation checklist.',
    },
    {
        num: '06',
        title: 'Report Generation & Export',
        href: '/report-generation',
        description:
            'Generate PDF reports for forensic documentation and export attack data for further analysis.',
    },
    {
        num: '07',
        title: 'User Roles & Authentication',
        href: '/user-roles',
        description:
            'Multi-user access with Admin & Analyst roles and secure login using OAuth2.',
    },
]

const Services = () => {
    return (
        <section className="min-h-[80vh] flex flex-col justify-center py-12 xl:py-0">
            <div className="container mx-auto">
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{
                        opacity: 1,
                        transition: {
                            delay: 2.4,
                            ease: 'easeIn',
                            duration: 0.4,
                        },
                    }}
                    className="grid grid-cols-1 md:grid-cols-2 gap-[72px] mt-16"
                >
                    {services.map((service, index) => {
                        return (
                            <div
                                key={index}
                                className={`flex flex-1 flex-col justify-center gap-6 group ${(index%2)?"animate-fromright":"animate-fromleft"}`}
                            >
                                <div className="w-full flex justify-between items-center">
                                    <div className="text-6xl font-extrabold text-outline text-transparent font-monospace group-hover:text-outline-hover transition-all duration-500">
                                        {service.num}
                                    </div>
                                    <Link
                                        href={service.href}
                                        className="w-[70px] h-[70px] rounded-full bg-foreground group-hover:bg-accent transition-all duration-500 flex justify-center items-center hover:-rotate-45"
                                    >
                                        <BsArrowDownRight className="text-background text-3xl" />
                                    </Link>
                                </div>
                                <h2 className="text-[36px] md:text-[42px] font-bold leading-none text-foreground group-hover:text-accent transition-all duration-500">
                                    {service.title}
                                </h2>
                                <p className="text-foreground/80">
                                    {service.description}
                                </p>
                                <div className="border-b border-foreground/20 w-full"></div>
                            </div>
                        )
                    })}
                </motion.div>
            </div>
        </section>
    )
}

export default Services
