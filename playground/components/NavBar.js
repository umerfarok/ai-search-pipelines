
import ChatBot from './ChatBot';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { useState } from 'react';
import { Menu, X, Search, Plus, Clock, Home, List, User } from 'lucide-react';

export default function NavBar() {
    const [isOpen, setIsOpen] = useState(false);
    const router = useRouter();

    const navItems = [
        { href: '/search', label: 'Search Models', icon: Search },
        { href: '/create-config', label: 'Create Config', icon: Plus },
        { href: '/pending-jobs', label: 'Pending Jobs', icon: Clock },
        { href: '/append-products', label: 'Append Products', icon: Plus },
    ];

    const isActive = (path) => router.pathname === path;

    return (
        <nav className="bg-gray-800 shadow-lg">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    {/* Logo/Brand */}
                    <div className="flex items-center">
                        <Link href="/" className="flex items-center text-white font-bold text-xl">
                            PlayGround
                        </Link>
                    </div>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center space-x-4">
                        {navItems.map(({ href, label, icon: Icon }) => (
                            <Link
                                key={href}
                                href={href}
                                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors
                                    ${isActive(href)
                                        ? 'bg-gray-900 text-white'
                                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                                    }`}
                            >
                                <Icon className="h-4 w-4 mr-2" />
                                {label}
                            </Link>
                        ))}
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex md:hidden items-center">
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 
                                hover:text-white hover:bg-gray-700 focus:outline-none focus:ring-2 
                                focus:ring-inset focus:ring-white"
                        >
                            {isOpen ? (
                                <X className="h-6 w-6" />
                            ) : (
                                <Menu className="h-6 w-6" />
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Menu */}
            <div className={`md:hidden ${isOpen ? 'block' : 'hidden'}`}>
                <div className="px-2 pt-2 pb-3 space-y-1">
                    {navItems.map(({ href, label, icon: Icon }) => (
                        <Link
                            key={href}
                            href={href}
                            className={`flex items-center px-3 py-2 rounded-md text-base font-medium
                                ${isActive(href)
                                    ? 'bg-gray-900 text-white'
                                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                                }`}
                            onClick={() => setIsOpen(false)}
                        >
                            <Icon className="h-5 w-5 mr-3" />
                            {label}
                        </Link>
                    ))}
                </div>

            </div>
            <ChatBot />
        </nav>
    );
}