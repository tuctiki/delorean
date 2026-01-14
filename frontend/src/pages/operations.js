import { useState, useEffect, useRef } from 'react';
import useSWR from 'swr';
import Link from 'next/link';
import { Play, Square, Terminal, Activity, FlaskConical, RefreshCw, CheckCircle, XCircle, Clock } from 'lucide-react';
import styles from '../styles/Operations.module.css';

const fetcher = (...args) => fetch(...args).then(res => res.json());

export default function Operations() {
    // Daily Task State
    const { data: dailyStatus } = useSWR('http://localhost:8000/api/status', fetcher, { refreshInterval: 1000 });
    const [dailyRunning, setDailyRunning] = useState(false);

    // Backtest State
    const { data: backtestStatus } = useSWR('http://localhost:8000/api/backtest-status', fetcher, { refreshInterval: 1000 });
    const [backtestRunning, setBacktestRunning] = useState(false);

    // Active log panel
    const [activeLog, setActiveLog] = useState('daily'); // 'daily' or 'backtest'
    const logRef = useRef(null);

    // Sync running states
    useEffect(() => {
        if (dailyStatus) setDailyRunning(dailyStatus.running);
    }, [dailyStatus]);

    useEffect(() => {
        if (backtestStatus) setBacktestRunning(backtestStatus.running);
    }, [backtestStatus]);

    // Auto-scroll log
    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [dailyStatus?.log, backtestStatus?.log]);

    const handleRunDaily = async () => {
        try {
            await fetch('http://localhost:8000/api/run-daily', { method: 'POST' });
            setActiveLog('daily');
        } catch (e) {
            console.error(e);
        }
    };

    const handleRunBacktest = async () => {
        try {
            await fetch('http://localhost:8000/api/run-backtest', { method: 'POST' });
            setActiveLog('backtest');
        } catch (e) {
            console.error(e);
        }
    };

    const currentLog = activeLog === 'daily' ? dailyStatus?.log : backtestStatus?.log;
    const currentRunning = activeLog === 'daily' ? dailyRunning : backtestRunning;

    const getStatusIcon = (running, exitCode) => {
        if (running) return <Clock size={16} className={styles.spinning} />;
        if (exitCode === 0 || exitCode === undefined) return <CheckCircle size={16} color="#2ecc71" />;
        return <XCircle size={16} color="#e74c3c" />;
    };

    return (
        <div className={styles.container}>
            {/* Header */}
            <div className={styles.header}>
                <h1><Terminal size={28} /> Operations Center</h1>
                <Link href="/" className={styles.backLink}>← Dashboard</Link>
            </div>

            <div className={styles.grid}>
                {/* Task Cards */}
                <div className={styles.taskCards}>
                    {/* Daily Task Card */}
                    <div className={`${styles.taskCard} ${activeLog === 'daily' ? styles.active : ''}`}
                        onClick={() => setActiveLog('daily')}>
                        <div className={styles.taskHeader}>
                            <div className={styles.taskIcon}>
                                <Activity size={24} />
                            </div>
                            <div className={styles.taskInfo}>
                                <h3>Daily Signal Generation</h3>
                                <p>Generate trading recommendations</p>
                            </div>
                            {getStatusIcon(dailyRunning)}
                        </div>
                        <div className={styles.taskStatus}>
                            {dailyRunning ? (
                                <span className={styles.running}>Running...</span>
                            ) : (
                                <span className={styles.ready}>Ready</span>
                            )}
                        </div>
                        <button
                            className={styles.runButton}
                            onClick={(e) => { e.stopPropagation(); handleRunDaily(); }}
                            disabled={dailyRunning}
                        >
                            {dailyRunning ? <><RefreshCw size={16} className={styles.spinning} /> Running</> : <><Play size={16} /> Run</>}
                        </button>
                    </div>

                    {/* Backtest Card */}
                    <div className={`${styles.taskCard} ${activeLog === 'backtest' ? styles.active : ''}`}
                        onClick={() => setActiveLog('backtest')}>
                        <div className={styles.taskHeader}>
                            <div className={styles.taskIcon} style={{ background: 'rgba(210, 168, 255, 0.1)' }}>
                                <FlaskConical size={24} color="#d2a8ff" />
                            </div>
                            <div className={styles.taskInfo}>
                                <h3>Full Backtest</h3>
                                <p>Train model and run historical backtest</p>
                            </div>
                            {getStatusIcon(backtestRunning, backtestStatus?.exit_code)}
                        </div>
                        <div className={styles.taskStatus}>
                            {backtestRunning ? (
                                <span className={styles.running}>Running...</span>
                            ) : backtestStatus?.exit_code === 0 ? (
                                <span className={styles.success}>Completed</span>
                            ) : backtestStatus?.exit_code ? (
                                <span className={styles.error}>Error</span>
                            ) : (
                                <span className={styles.ready}>Ready</span>
                            )}
                        </div>
                        <button
                            className={`${styles.runButton} ${styles.purple}`}
                            onClick={(e) => { e.stopPropagation(); handleRunBacktest(); }}
                            disabled={backtestRunning}
                        >
                            {backtestRunning ? <><RefreshCw size={16} className={styles.spinning} /> Running</> : <><Play size={16} /> Run</>}
                        </button>
                    </div>
                </div>

                {/* Log Monitor */}
                <div className={styles.logPanel}>
                    <div className={styles.logHeader}>
                        <div className={styles.logTabs}>
                            <button
                                className={`${styles.logTab} ${activeLog === 'daily' ? styles.activeTab : ''}`}
                                onClick={() => setActiveLog('daily')}
                            >
                                <Activity size={14} /> Daily Task
                                {dailyRunning && <span className={styles.liveDot}></span>}
                            </button>
                            <button
                                className={`${styles.logTab} ${activeLog === 'backtest' ? styles.activeTab : ''}`}
                                onClick={() => setActiveLog('backtest')}
                            >
                                <FlaskConical size={14} /> Backtest
                                {backtestRunning && <span className={styles.liveDot}></span>}
                            </button>
                        </div>
                        <div className={styles.logStatus}>
                            {currentRunning ? (
                                <span className={styles.liveIndicator}>● LIVE</span>
                            ) : (
                                <span className={styles.idleIndicator}>○ IDLE</span>
                            )}
                        </div>
                    </div>
                    <div className={styles.logContent} ref={logRef}>
                        {currentLog ? (
                            <pre>{currentLog}</pre>
                        ) : (
                            <div className={styles.logPlaceholder}>
                                <Terminal size={48} strokeWidth={1} />
                                <p>No log output yet. Run a task to see output here.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
