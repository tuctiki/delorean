import Link from 'next/link';
import { useRouter } from 'next/router';
import { Home, BarChart2, Database, Github } from 'lucide-react';
import styles from '../styles/Navbar.module.css';

export default function Navbar() {
    const router = useRouter();

    const isActive = (path) => router.pathname === path;

    return (
        <nav className={styles.navbar}>
            <div className={styles.logo}>
                <span className={styles.logoText}>Delorean</span>
            </div>
            <div className={styles.links}>
                <Link href="/" className={`${styles.link} ${isActive('/') ? styles.active : ''}`}>
                    <Home size={18} />
                    <span>Dashboard</span>
                </Link>
                <Link href="/data" className={`${styles.link} ${isActive('/data') ? styles.active : ''}`}>
                    <Database size={18} />
                    <span>Data</span>
                </Link>
                <Link href="/experiments" className={`${styles.link} ${isActive('/experiments') ? styles.active : ''}`}>
                    <BarChart2 size={18} />
                    <span>Experiments</span>
                </Link>
            </div>
        </nav>
    );
}
