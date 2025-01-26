import Link from 'next/link';

export default function NavBar() {
    return (
        <nav className="bg-gray-800 p-4">
            <div className="container mx-auto flex justify-between">
                <Link href="/" className="text-white">
                    Home
                </Link>
                <Link href="/search" className="text-white">
                    Search Through Models
                </Link>
                <Link href="/create-config" className="text-white">
                    Create Train Config
                </Link>
                <Link href="/pending-jobs" className="text-white">
                    Pending Jobs
                </Link>
            </div>
        </nav>
    );
}