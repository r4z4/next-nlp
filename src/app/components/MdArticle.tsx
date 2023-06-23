import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import jsx from 'react-syntax-highlighter/dist/esm/languages/prism/jsx';
import js from 'react-syntax-highlighter/dist/esm/languages/prism/javascript';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import { dark } from "react-syntax-highlighter/dist/esm/styles/prism";
import SidePanelImageDisplay from './SidePanelImageDisplay'
import style from './markdown-styles.module.css';
import dfHtml from '../utils/Dataframes'

SyntaxHighlighter.registerLanguage('jsx', jsx);
SyntaxHighlighter.registerLanguage('javascript', js);
SyntaxHighlighter.registerLanguage('python', python);

interface MdArticleProps {
    title: string;
    subDir: string;
}

function MdArticle({ title, subDir }: MdArticleProps) {
    const mdPath = require(`../articles/${subDir}/${title}.md`)
    const [terms, setTerms] = React.useState('')
    const [html, setHtml] = React.useState('')

    const [expanded, setExpanded] = React.useState(false);

    const imagesPath = `../assets/article_images/${subDir}/${title}/`

    React.useEffect(() => {
        if (subDir === 'glove') { 
          if (title === 'run_03') {
            setHtml(dfHtml['run3html']) 
          }
          if (title === 'run_04') {
            setHtml(dfHtml['run4html']) 
          }
        }
        if (subDir === 'topic-modeling') { 
          if (title === '01_transformers') {
            setHtml(dfHtml['tm_1_html_1'] + "<br />" + dfHtml['tm_1_html_2'] + "<br />" + dfHtml['tm_1_html_3'] + "<br />" + dfHtml['tm_1_html_4']) 
          }
        }
        if (subDir === 'trec') { 
          if (title === 'trec_eda') {
            setHtml(dfHtml['trec_eda_1'] + "<br />" + dfHtml['trec_eda_2'] + "<br />" + dfHtml['trec_eda_3'] + "<br />" + dfHtml['trec_eda_4'] + "<br />" + dfHtml['trec_eda_5']) 
          }
        }
        if (subDir === 'trivia') { 
          if (title === 'lda_trivia') {
            setHtml(dfHtml['lda_trivia_1'] + "<br />" + dfHtml['lda_trivia_2'] + "<br />" + dfHtml['lda_trivia_3'] + "<br />" + dfHtml['lda_trivia_4']) 
          }
        }
        fetch(mdPath).then((response) => response.text()).then((text) => {
            
            setTerms(text)
        })
        // setTerms(markdownTable)
    }, [mdPath, subDir, title])

    return (
      <div className='grid-container'>
        <div>
          <button className="toggle-button" onClick={() => setExpanded(!expanded)}>{expanded ? '>' : '<' }</button>
          <ReactMarkdown 
            className={style.reactMarkDown} 
            children={terms}
            remarkPlugins={[[remarkGfm, {tableCellPadding: true, tablePipeAlign: true}]]}
            components={{
              code({node, inline, className, children, ...props}) {
                const match = /language-(\w+)/.exec(className || '')
                return !inline && match ? (
                  <SyntaxHighlighter
                    {...props}
                    children={String(children).replace(/\n$/, '')}
                    style={dark}
                    wrapLines={true}
                    language={match[1]}
                    PreTag="div"
                  />
                ) : (
                  <code {...props} className={className}>
                    {children}
                  </code>
                )
              }
            }}/>
        </div>
        {expanded ? (
          <div className='flex-container'>
            <SidePanelImageDisplay html={html} imagesPath={imagesPath} />
          </div>  
          ) : null
        }
      </div>
    )
  }

export default MdArticle