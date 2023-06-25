"use client"
import React from 'react'
import Image from "next/image";
import FolderClosedIcon from '../assets/folder_closed.svg'
import FolderOpenIcon from '../assets/folder_open.svg'
import NotebookSimple from '../assets/notebook_simple.svg'
import Article, { ArticleProps } from "@/app/components/Article";

export interface CollapsePanelProps {
    panelData: PanelData;
}

// export interface PanelDocument {
//     id: number;
//     filename: String;
//     url: string;
//     previewComponent: JSX.Element;
// }

export interface PanelData {
    name: String;
    date: String;
    desc: String;
    img?: string;
    bgColor?: string;
    category: String;
    documents: ArticleProps[];
}

// const panelData {
//     name = 'Panel Name',
//     date = '05-22-2020',
//     category = 'Government',
//     documents = ['Doc1', 'Doc2', 'Doc3']
// }

function CollapsePanel({ panelData }: CollapsePanelProps) {
  const [expanded, setExpanded] = React.useState(false);
  // const [isShown, setIsShown] = React.useState<JSX.Element>(<></>);

  return (
    <div className="collapse-panel" style={{backgroundColor: panelData.bgColor ? panelData.bgColor : ''}}>
      <span className="show-more" onClick={() => setExpanded(!expanded)}>
        <div className="collapse-img">
          {expanded ? <Image width={35} className="white-filter" src={FolderOpenIcon} alt='folderClosedIcon'/> : <Image width={35} className="white-filter" src={FolderClosedIcon} alt='folderOpenIcon'/>}
        </div>
        <p className='panel-dir-name'>{panelData.name}</p>
      </span>
        <p>Last Updated: {panelData.date}</p>
        <p>{panelData.desc}</p>
        <p>Category = {panelData.category}</p>
      {expanded ? (
        <div className="expandable">
          {panelData.documents.map((doc: ArticleProps) => (
            <div className="dir-grid">
              <ul className='panel-doc-list'>
                <div className='file-grid'>
                  <li 
                    key={doc.id}>
                    {/* onMouseEnter={() => setIsShown(doc.previewComponent)} onMouseLeave={() => setIsShown(<></>)}> */}
                    
                      <a className="text-gradient-ash" href={`/articles/${doc.id}`}><Image className="icon white-filter" src={NotebookSimple} alt='notebookSimpleIcon'/>{doc.filename}</a>
                  </li>
                </div>
              </ul>
          </div>
          ))}
        </div>
        ) : null
      } 
      {/*{isShown ? (<div className='prev-div' key='prev'>{isShown}</div>) : null}*/}
    </div>
  );
}

export default CollapsePanel;