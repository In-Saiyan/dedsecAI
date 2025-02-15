'use client'
import React from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { FaPhoneAlt, FaEnvelope, FaMapMarkerAlt } from 'react-icons/fa'
import { motion } from 'framer-motion'

const contactInfo = [
    {
        icon: <FaPhoneAlt />,
        title: 'Phone',
        value: '+91 1234567890',
    },
    {
        icon: <FaEnvelope />,
        title: 'Email',
        value: 'dedsecaiapp@gmail.com',
    },
    {
        icon: <FaMapMarkerAlt />,
        title: 'Address',
        value: 'Dedsec AI, 123, Example Street, Lucknow, UP',
    },
]

const Contact = () => {
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
                    <div className="xl:w-[64%] order-2 xl:order-none">
                        <form className="flex flex-col gap-6 p-10 bg-secondary rounded-xl font-monospace">
                            <h3 className="text-4xl text-accent font-bold">
                                Contact Us
                            </h3>
                            <p className="text-foreground/60">
                                We&apos;re here to help and answer any question
                                you might have. We look forward to hearing from
                                you.
                            </p>
                            <div className="grid grid-cols-1 gap-6">
                                <Input type="text" placeholder="First Name" />
                                <Input type="text" placeholder="Last Name" />
                                <Input
                                    type="email"
                                    placeholder="Email Address"
                                />
                                <Input
                                    type="phone"
                                    placeholder="Phone Number"
                                />
                            </div>
                            <Textarea
                                className={`h-[120px]`}
                                placeholder="Type your message here."
                            />
                            <Button className="max-w-40">Send Message</Button>
                        </form>
                    </div>
                    <div
                        className="flex-1 flex items-center xl:justify-center order-1
                    xl:order-none mb-8 xl:mb-0 flex-col"
                    >
                        <h1 className="font-black text-3xl mb-3 text-left pl-8 w-full font-monospace">
                            Info
                        </h1>
                        <ul className="flex flex-col gap-10 p-5 font-monospace">
                            {contactInfo.map((info, index) => {
                                return (
                                    <li
                                        key={index}
                                        className="flex gap-6 items-center"
                                    >
                                        <div className="w-[52px] h-[52px] xl:w-[52px] xl:h-[52px] rounded-md bg-secondary text-accent flex items-center justify-center aspect-square">
                                            <div className="text-[28px]">
                                                {info.icon}
                                            </div>
                                        </div>
                                        <div className="flex flex-col gap-2">
                                            <p className="text-sm text-foreground/70">
                                                {info.title}
                                            </p>
                                            <h3 className="text-foreground leading-6">
                                                {info.value}
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

export default Contact
