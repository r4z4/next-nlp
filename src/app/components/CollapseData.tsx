"use client"
import React from 'react'
import { useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import FolderClosedIcon from '../assets/folder_closed.svg'
import FolderOpenIcon from '../assets/folder_open.svg'
import NotebookSimple from '../assets/notebook_simple.svg'
import Article, { ArticleProps } from "@/app/components/Article";
import CollapsePanel from "../components/CollapsePanel";

export interface CollapsePanelProps {
    panelData: PanelData;
}

interface CategoryProps {
  name: string
  articles: ArticleProps[]
}

export interface PanelData {
    name: String;
    date: String;
    desc: String;
    img?: string;
    bgColor?: string;
    category: String;
    documents: PanelDocument[];
}

export interface PanelDocument {
    id: number;
    filename: String;
    url: string;
    previewComponent: JSX.Element;
}

function getPanelData(category: CategoryProps) {
  if (category.name == 'trivia') {
    return {
        name: "Test Run Trivia",
        date: "05/30/2020",
        desc: "Test Desc",
        img: '',
        bgColor: "#FFFFFF",
        category: "trivia",
        documents: []
    }
  }
  if (category.name == 'news') {
    return {
      name: "Test Run News",
      date: "05/30/2020",
      desc: "Test Desc",
      img: '',
      bgColor: "#FFFFFF",
      category: "news",
      documents: []
    }
  }
}


// const panelData {
//     name = 'Panel Name',
//     date = '05-22-2020',
//     category = 'Government',
//     documents = ['Doc1', 'Doc2', 'Doc3']
// }

function CollapseClient({ data }: any) {
  const [expanded, setExpanded] = React.useState(false);
  const [isShown, setIsShown] = React.useState<JSX.Element>(<></>);

  useEffect(() => {
    console.log(`useEffect Firing & Data = ${JSON.stringify(data)}`)
  } 
  , [data]);

  return (
      <>
        <ul>
            {data?.map((category: CategoryProps) => (
                <li>
                    <CollapsePanel panelData={getPanelData(category)} /> 
                </li>
            ))}
        </ul>
    </>
  );
}

export default CollapseClient;