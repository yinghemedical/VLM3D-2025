# -*- coding: utf-8 -*-
"""
RadBert 报告分类提取器

使用预训练的 RadBert 分类器从放射学报告中提取多标签分类结果。
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModel


# 默认的 RoBERTa 基础模型路径
DEFAULT_ROBERTA_DIR = "/path/to/robert"

# CT-RATE 18 类病理标签
CTRATE_CLS_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening"
]


class ReportDataset(Dataset):
    """报告数据集，用于 RadBert 推理"""

    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str,
        report_col: str,
        tokenizer,
        max_length: int = 512
    ):
        self.df = df
        self.id_col = id_col
        self.report_col = report_col
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row[self.id_col])
        report = str(row[self.report_col]) if pd.notna(row[self.report_col]) else " "

        encoding = self.tokenizer(
            report,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sample_id': sample_id
        }


class RadBertClassifier(torch.nn.Module):
    """RadBert 分类器模型

    使用 RoBERTa 作为基础模型，与 CT-RATE 官方评估代码一致。
    """

    def __init__(self, n_classes: int = 18, base_model_dir: str = None):
        super().__init__()

        model_dir = Path(base_model_dir or DEFAULT_ROBERTA_DIR).resolve()
        assert model_dir.exists(), f"Base model not found: {model_dir}"

        self.config = AutoConfig.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir, config=self.config)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        return output


class RadBertExtractor:
    """
    RadBert 分类提取器

    从报告文本中提取多标签分类结果。
    """

    def __init__(
        self,
        model_path: str,
        cls_cols: Optional[List[str]] = None,
        base_model_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        初始化提取器

        Args:
            model_path: RadBert 分类器权重路径 (.pth)
            cls_cols: 分类标签列表
            base_model_dir: RoBERTa 基础模型目录
            device: 设备 ('cuda' 或 'cpu')
            batch_size: 批处理大小
        """
        self.model_path = model_path
        self.cls_cols = cls_cols or CTRATE_CLS_COLS
        self.base_model_dir = base_model_dir or DEFAULT_ROBERTA_DIR
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_dir,
            do_lower_case=True,
            local_files_only=True
        )

        self.model = self._load_model()
        print(f"RadBert model loaded: {len(self.cls_cols)} classes, device={self.device}")

    def _load_model(self) -> RadBertClassifier:
        """加载模型"""
        assert Path(self.model_path).exists(), f"Model not found: {self.model_path}"

        model = RadBertClassifier(
            n_classes=len(self.cls_cols),
            base_model_dir=self.base_model_dir
        )
        state_dict = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()

        return model

    def extract(
        self,
        df: pd.DataFrame,
        id_col: str,
        report_col: str
    ) -> pd.DataFrame:
        """
        从报告中提取分类结果

        Args:
            df: 包含报告的 DataFrame
            id_col: ID 列名
            report_col: 报告列名

        Returns:
            包含 ID 和分类预测的 DataFrame
        """
        dataset = ReportDataset(df, id_col, report_col, self.tokenizer)
        loader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

        all_probs = []
        all_ids = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.append(probs)
                all_ids.extend(batch['sample_id'])

        all_probs = np.vstack(all_probs)

        # 构建结果 DataFrame
        result_df = pd.DataFrame(all_probs, columns=self.cls_cols)
        result_df.insert(0, id_col, all_ids)

        return result_df

    def extract_binary(
        self,
        df: pd.DataFrame,
        id_col: str,
        report_col: str,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        从报告中提取二值化的分类结果

        Args:
            df: 包含报告的 DataFrame
            id_col: ID 列名
            report_col: 报告列名
            threshold: 二值化阈值

        Returns:
            包含 ID 和二值化分类预测的 DataFrame
        """
        result_df = self.extract(df, id_col, report_col)

        # 二值化
        for col in self.cls_cols:
            result_df[col] = (result_df[col] >= threshold).astype(int)

        return result_df
