import { prisma } from "@/app/db";
import styles from './Button.module.css'
import formStyles from './Form.module.css'


export default async function ArticleEditPage({
    params,
}: {
    params: {id: number };
}) {
    const key = `articles:${params.id}`;
    const article = await prisma.article.findUnique({
        where: {
          id: params.id,
        },
      });

    async function upArticle(formData: FormData) {
        "use server";
        await prisma.article.update({where: {
                id: params.id
            },
            data: {
                title: formData.get("title") as string || '',
                category: formData.get("category") as string || '',
                published: formData.get("publihsed") as unknown || false,
            }
          })
    }

    return (
        <div className={formStyles.card}>
            <div className={formStyles.cardBody}>
                <h2>Edit {article?.title}</h2>
                <form action={upArticle}>
                    <label>Name</label>
                    <input 
                        name="title"     
                        required
                        minLength={10}
                        maxLength={20}
                        type="text" 
                        defaultValue={article?.title} 
                    />
                    <label>Image</label>
                    <input name="title" type="text" defaultValue={article?.category} />
                    <label>Complete</label>
                    <input name="title" type="checkbox" checked={article?.published} />

                    <button className={styles.error} type="submit"> Save & Continue</button>
                    
                </form>
            </div>
        </div>
    )
}