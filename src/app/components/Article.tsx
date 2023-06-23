"use client"
import React from 'react'
import { useRouter } from 'next/navigation'
import styles from '../components/Article.module.css'

export type ArticleProps = {
  id: number
  title: string
  published: boolean
  createdAt: Date
  updatedAt: Date
}

const Post: React.FC<{ article: ArticleProps }> = ({ article }) => {
  const router = useRouter()
  const articleTitle = article.title ? article.title : 'Unknown author'
  return (
    <div
      className={styles.post}
      onClick={() => router.push('/p/[id]', `/p/${article.id}`)}
    >
      <h2>{article.title}</h2>
      <small>By {article.title}</small>
    </div>
  )
}

export default Post