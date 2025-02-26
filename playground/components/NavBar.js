import ChatBot from './ChatBot';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { useState, useEffect } from 'react';
import { Menu, X, Search, Plus, Clock, Home, List, User, Sun, Moon } from 'lucide-react';
import { useTheme } from 'next-themes';

export default function NavBar() {
    const [isOpen, setIsOpen] = useState(false);
    const router = useRouter();
    const { theme, setTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    const navItems = [ 
        { href: '/search', label: 'Search Models', icon: Search },
        { href: '/create-config', label: 'Create Config', icon: Plus },
        { href: '/pending-jobs', label: 'Pending Jobs', icon: Clock },
        { href: '/append-products', label: 'Append Products', icon: Plus },
    ];

    const isActive = (path) => router.pathname === path;

    const renderThemeToggle = () => {
        if (!mounted) return null;
        
        return (
            <button
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                className="p-2 rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800/60 transition-colors"
                aria-label="Toggle theme"
            >
                {theme === 'dark' ? (
                    <Sun className="h-5 w-5" />
                ) : (
                    <Moon className="h-5 w-5" />
                )}
            </button>
        );
    };

    return (
        <nav className="sticky top-0 z-50 w-full border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shadow-sm">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    {/* Logo/Brand */}
                    <div className="flex items-center">
                        <Link href="/" className="flex items-center">
                            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent dark:from-blue-400 dark:to-indigo-400">
                                PlayGround
                            </span>
                        </Link>
                    </div>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center space-x-4">
                        {navItems.map(({ href, label, icon: Icon }) => (
                            <Link
                                key={href}
                                href={href}
                                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ease-in-out
                                    ${isActive(href)
                                        ? 'bg-gray-100 dark:bg-gray-800 text-blue-600 dark:text-blue-400'
                                        : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800/60 hover:text-blue-600 dark:hover:text-blue-400'
                                    }`}
                            >
                                <Icon className="h-4 w-4 mr-2" />
                                {label}
                            </Link>
                        ))}

                        {/* Theme Toggle */}
                        {renderThemeToggle()}
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex md:hidden items-center space-x-2">
                        {renderThemeToggle()}
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="inline-flex items-center justify-center p-2 rounded-md text-gray-600 dark:text-gray-300
                                hover:bg-gray-100 dark:hover:bg-gray-800"
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
            <div className={`md:hidden transition-all duration-200 ease-in-out ${isOpen ? 'block' : 'hidden'}`}>
                <div className="px-2 pt-2 pb-3 space-y-1 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800">
                    {navItems.map(({ href, label, icon: Icon }) => (
                        <Link
                            key={href}
                            href={href}
                            className={`flex items-center px-3 py-2 rounded-md text-base font-medium
                                ${isActive(href)
                                    ? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white'
                                    : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white'
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