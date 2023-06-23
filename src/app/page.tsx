"use client"
import Link from "next/link";

export default function Home() {
  return  (
    <header>
      <h1 className="header-title">Hello</h1>
      <Link href="/news">News</Link>
    </header>
  )}