import {
    FaGithub,
    FaFacebook,
    FaInstagram,
    FaTwitter,
    FaLinkedin,
} from 'react-icons/fa'

export const mainNavLinks = [
    { name: 'Home', path: '/' },
    { name: 'Services', path: '/services' },
    { name: 'About', path: '/about' },
    { name: 'Contact', path: '/contact' },
]

export const footerLinks = [
    {
        title: 'Application',
        links: [
            { title: 'Dashboard', url: '/dashboard' },
            { title: 'Services', url: '/services' },
            { title: 'Report Generation', url: '/report-generation' },
            { title: 'Log Analyzer', url: '/log-analyzer' },
        ],
    },
    {
        title: 'Company',
        links: [
            { title: 'About Us', url: '/about' },
            { title: 'Contact Us', url: '/contact' },
            { title: 'Privacy Policy', url: '/privacy' },
            { title: 'Terms & Conditions', url: '/terms' },
        ],
    },
]

export const socialLinks = [
    {
        title: 'Github',
        icon: FaGithub,
        url: '/',
    },
    {
        title: 'Instagram',
        icon: FaInstagram,
        url: '/',
    },
    {
        title: 'Twitter',
        icon: FaTwitter,
        url: '/',
    },
    {
        title: 'Facebook',
        icon: FaFacebook,
        url: '/',
    },
    {
        title: 'LinkedIn',
        icon: FaLinkedin,
        url: '/',
    },
]
