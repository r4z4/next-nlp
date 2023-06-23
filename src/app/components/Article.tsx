import React from 'react'
import Router from 'next/router'
import styles from '@/components/Post.module.css'

export type ArticleProps = {
  id: string
  title: string
  complete: boolean
  createdAt: Date
  updatedAt: Date
}

const Post: React.FC<{ article: ArticleProps }> = ({ article }) => {
  const articleTitle = article.title ? article.title : 'Unknown author'
  return (
    <div
      className={styles.post}
      onClick={() => Router.push('/p/[id]', `/p/${article.id}`)}
    >
      <h2>{article.title}</h2>
      <small>By {article.title}</small>
    </div>
  )
}

export default Post