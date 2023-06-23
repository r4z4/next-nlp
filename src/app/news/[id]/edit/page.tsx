import db from '@/app/db';
import styles from './Button.module.css'
import formStyles from './Form.module.css'

export default async function ArticleEditPage({
    params,
}: {
    params: {id: number };
}) {
    const key = `articles:${params.id}`;
    const article = await db.run(`INSERT INTO`);

    async function upArticle(formData: FormData) {
        "use server";
        db.run(`INSERT INTO`)
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