"use client"
import Article, { BlogTitle } from './run_02.mdx';

export default async function NewsArticle() {

    return (
        <div>
            <p>Hey</p>
            <Article />;
            <BlogTitle />;
        </div>
    )
}