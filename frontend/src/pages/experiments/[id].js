import { useRouter } from 'next/router';
import useSWR from 'swr';
import { ArrowLeft, Box, List, TrendingUp } from 'lucide-react';
import Link from 'next/link';
import styles from '../../styles/ExperimentDetail.module.css';

const fetcher = (...args) => fetch(...args).then(res => res.json());

export default function ExperimentDetail() {
    const router = useRouter();
    const { id } = router.query;

    const { data: experiment, error } = useSWR(id ? `http://localhost:8000/api/experiments/${id}` : null, fetcher);

    if (error) return <div className={styles.center}>Error loading experiment.</div>;
    if (!experiment) return <div className={styles.center}>Loading...</div>;

    return (
        <div className={styles.container}>
            <Link href="/experiments" className={styles.backLink}>
                <ArrowLeft size={16} /> Back to Experiments
            </Link>

            <h1 className={styles.title}>Experiment #{id}</h1>

            <div className={styles.grid}>
                {/* Params Card */}
                <div className={styles.card}>
                    <div className={styles.cardHeader}>
                        <Box size={20} /> Parameters
                    </div>
                    <div className={styles.tableWrapper}>
                        <table className={styles.table}>
                            <tbody>
                                {Object.entries(experiment.params || {}).map(([key, value]) => (
                                    <tr key={key}>
                                        <td className={styles.key}>{key}</td>
                                        <td className={styles.value}>{String(value)}</td>
                                    </tr>
                                ))}
                                {Object.keys(experiment.params || {}).length === 0 && (
                                    <tr><td colSpan="2" className={styles.empty}>No parameters logged</td></tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Metrics Card */}
                <div className={styles.card}>
                    <div className={styles.cardHeader}>
                        <TrendingUp size={20} /> Metrics
                    </div>
                    <div className={styles.tableWrapper}>
                        <table className={styles.table}>
                            <tbody>
                                {Object.entries(experiment.metrics || {}).map(([key, value]) => (
                                    <tr key={key}>
                                        <td className={styles.key}>{key}</td>
                                        <td className={styles.value}>{typeof value === 'number' ? value.toFixed(4) : value}</td>
                                    </tr>
                                ))}
                                {Object.keys(experiment.metrics || {}).length === 0 && (
                                    <tr><td colSpan="2" className={styles.empty}>No metrics logged</td></tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* Artifacts / Info */}
            <div className={styles.card}>
                <div className={styles.cardHeader}>
                    <List size={20} /> Info
                </div>
                <div className={styles.infoRow}>
                    <strong>Artifact Location:</strong>
                    <span className={styles.path}>{experiment.artifact_location}</span>
                </div>
                <div className={styles.infoRow}>
                    <strong>Status:</strong>
                    <span>{experiment.status || 'FINISHED'}</span>
                </div>
            </div>
        </div>
    );
}
