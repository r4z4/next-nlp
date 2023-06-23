import Article, { ArticleProps } from "@/app/components/Article";
import articleStyles from './Article.module.css'
import { prisma } from "@/app/db";

export default async function ArticlePage({
    params,
}: {
    params: {id: string };
}) {
    const key = `articles:${params.id}`;
    const articles = prisma.article.findMany({
        orderBy: [
          {
            id: 'desc',
          },
        ],
      })

    return (
        <div className={articleStyles.card}>
            <div className={articleStyles.cardBody}>
                <h2>Articles</h2>
                <div>
                    <ul>
                        {(await articles).map((article: ArticleProps) => (
                            <li>
                                <Article article={article} />
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    )
}