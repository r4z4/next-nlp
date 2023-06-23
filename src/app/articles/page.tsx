import Article, { ArticleProps } from "@/app/components/Article";
import homeStyles from '../components/Home.module.css'
import { prisma } from "@/app/db";

export default async function ArticleHome({
    params,
}: {
    params: {id: number };
}) {
    const articles = prisma.article.findMany({
        orderBy: [
          {
            id: 'desc',
          },
        ],
      })

    return (
        <div className={homeStyles.card}>
            <div className={homeStyles.cardBody}>
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