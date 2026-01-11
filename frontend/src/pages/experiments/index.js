import useSWR from 'swr';
import { Layers, FileText } from 'lucide-react';
import styles from '../../styles/Experiments.module.css';

const fetcher = (...args) => fetch(...args).then(res => res.json());

export default function Experiments() {
    const { data: experiments, error } = useSWR('http://localhost:8000/api/experiments', fetcher);

    return (
        <div className={styles.container}>
            <h1 className={styles.title}>Experiments</h1>

            <div className={styles.list}>
                {experiments?.map(exp => (
                    <div key={exp.id} className={styles.item}>
                        <div className={styles.icon}>
                            <Layers size={24} />
                        </div>
                        <div className={styles.content}>
                            <h3>Experiment #{exp.id}</h3>
                            <p className={styles.path}>{exp.artifact_location}</p>
                        </div>
                        <div className={styles.actions}>
                            <a href={`/experiments/${exp.id}`}>
                                <button className={styles.viewBtn}><FileText size={16} /> Details</button>
                            </a>
                        </div>
                    </div>
                ))}
                {experiments && experiments.length === 0 && <div className={styles.empty}>No experiments found in mlruns.</div>}
                {(!experiments && !error) && <div className={styles.loading}>Loading...</div>}
            </div>
        </div>
    );
}
