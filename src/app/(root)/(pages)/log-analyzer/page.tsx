'use client'
import React from 'react'

import systemLogs from '@/data/systemLogs.json'
import { motion } from 'framer-motion'

function formatDateTime(dateTimeStr: string) {
    const date = new Date(dateTimeStr);
    
    // Extract time components
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const seconds = date.getSeconds().toString().padStart(2, '0');
    
    // Extract date components
    const day = date.getDate();
    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const month = monthNames[date.getMonth()];
    const year = date.getFullYear();
    
    // Get ordinal suffix for the day
    const ordinalSuffix = (d: number) => {
        if (d > 3 && d < 21) return 'th'; // Covers 4th-20th
        switch (d % 10) {
            case 1: return 'st';
            case 2: return 'nd';
            case 3: return 'rd';
            default: return 'th';
        }
    };
    
    const formattedDate = `${hours}:${minutes}:${seconds} ${day}${ordinalSuffix(day)} ${month}, ${year}`;
    return formattedDate;
}

const LogAnalyzer = () => {
    return (
        <motion.section
            initial={{ opacity: 0 }}
            animate={{
                opacity: 1,
                transition: { delay: 2.4, ease: 'easeIn', duration: 0.4 },
            }}
            className="py-6"
        >
            <div className="container mx-auto">
                <div className="flex flex-col xl:flex-row gap-[30px]">
                    <div
                        className="flex-1 flex items-center xl:justify-center order-1
                    xl:order-none mb-8 xl:mb-0 flex-col gap-6 p-10 bg-secondary rounded-xl font-monospace"
                    >
                        <h1 className="font-black text-4xl mb-3 text-center pl-8 font-monospace">
                            Suspicious System Logs
                        </h1>
                        <ul className="flex flex-col gap-6 font-monospace">
                            {systemLogs.map((log, index) => {
                                return (
                                    <li
                                        key={index}
                                        className="flex flex-col gap-6 p-5 rounded-2xl items-start bg-background"
                                    >
                                        <div className="flex flex-col gap-4 items-start justify-center">
                                            <div className="text-lg leading-6">
                                                <span className='font-bold'>Time &rarr;&nbsp;</span>
                                                <span className="text-accent">{formatDateTime(log.timestamp)}</span>
                                            </div>
                                            <h3 className="text-foreground/70 font-primary  leading-7 tracking-wide">
                                                {log.message}
                                            </h3>
                                        </div>
                                    </li>
                                )
                            })}
                        </ul>
                    </div>
                </div>
            </div>
        </motion.section>
    )
}

export default LogAnalyzer
