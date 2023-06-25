"use client"
import Article, { BlogTitle } from './run_02.mdx';
import Md from '../assets/md/glove/run_03.md'
import Mdx from '../assets/mdx/glove/run_04.mdx'
import ArticleImage from '../assets/article_images/glove/run_04.png'

export default async function NewsArticle() {

    return (
        <div>
            <p>Hey</p>
            <Mdx />;
            <BlogTitle />;
        </div>
    )
}