"use server";
import db from '@/app/db';
import Article, { ArticleProps } from "@/app/components/Article";

interface CategoryProps {
    [x: string]: any;
    articles: ArticleProps[]
}

export async function getArticles(): Promise<CategoryProps> {
    console.log('wtf')
    return db.all(`SELECT * FROM articles;`, [], function (err: any, row: ArticleProps) {
        if (err) {
            throw err;
        }
        return row;
    }); 
}