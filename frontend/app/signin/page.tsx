"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { API_BASE, getAuthProfile } from "@/lib/api";
import styles from "./signin.module.css";

export default function SignInPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAuthProfile()
      .then(() => router.replace("/upload"))
      .catch(() => setLoading(false));
  }, [router]);

  if (loading) {
    return <div className={styles.loading}>checking session...</div>;
  }

  return (
    <div className={styles.page}>
      <nav className={styles.nav}>
        <Link href="/" className={styles.logo}>
          <span className={styles.logoDot} />
          DeepRead
        </Link>
        <Link href="/" className={styles.navBack}>
          {"<-"} back to home
        </Link>
      </nav>

      <div className={styles.content}>
        <div className={styles.card}>
          <h1 className={styles.heading}>Sign in to continue</h1>

          <a href={`${API_BASE}/auth/google`} className={styles.btnGoogle}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M15.68 8.18c0-.57-.05-1.12-.14-1.64H8v3.1h4.3a3.68 3.68 0 0 1-1.6 2.42v2h2.58c1.51-1.4 2.4-3.45 2.4-5.88Z" fill="#fff" fillOpacity=".8" />
              <path d="M8 16c2.16 0 3.97-.72 5.3-1.94l-2.59-2a4.8 4.8 0 0 1-7.15-2.52H.88v2.07A8 8 0 0 0 8 16Z" fill="#fff" fillOpacity=".6" />
              <path d="M3.56 9.54A4.8 4.8 0 0 1 3.3 8c0-.54.09-1.06.25-1.54V4.39H.88A8 8 0 0 0 0 8c0 1.29.31 2.51.88 3.61l2.68-2.07Z" fill="#fff" fillOpacity=".4" />
              <path d="M8 3.18c1.22 0 2.3.42 3.16 1.24l2.37-2.37A8 8 0 0 0 .88 4.39l2.68 2.07A4.77 4.77 0 0 1 8 3.18Z" fill="#fff" fillOpacity=".9" />
            </svg>
            Continue with Google
          </a>

          <a href={`${API_BASE}/auth/github`} className={styles.btnGithub}>
            <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
            </svg>
            Continue with GitHub
          </a>

          <Link href="/" className={styles.backLink}>
            {"<-"} back to home
          </Link>
        </div>
      </div>
    </div>
  );
}
