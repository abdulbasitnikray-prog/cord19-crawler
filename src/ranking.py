"""
ranking.py - Comprehensive ranking system for CORD-19 medical papers
Simplified version without explanation functions.
"""

import math
import re
import pickle
import os
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np

# ============================================================================
# AUTHOR PROMINENCE ANALYZER (Data-Driven)
# ============================================================================

class AuthorProminenceAnalyzer:
    """Automatically identifies prominent authors from the dataset."""
    
    def __init__(self, metadata_df: pd.DataFrame, citation_builder=None):
        self.metadata_df = metadata_df
        self.citation_builder = citation_builder
        self.author_stats = {}
        if not self.load_analysis():
            self._analyze_authors()
    
    def _analyze_authors(self):
        """Analyze all authors in the dataset."""
        # Use regular dicts for better memory efficiency
        author_papers = {}
        author_journals = {}
        author_years = {}
        
        for _, row in self.metadata_df.iterrows():
            paper_id = str(row['cord_uid']).strip()
            journal = str(row.get('journal', '')).lower()
            authors_str = str(row.get('authors', ''))
            publish_time = str(row.get('publish_time', ''))
            
            if not authors_str or authors_str.lower() == 'nan':
                continue
            
            authors = self._parse_authors(authors_str)
            
            for author in authors:
                author_lower = author.lower().strip()
                author_papers.setdefault(author_lower, []).append(paper_id)
                if journal:
                    author_journals.setdefault(author_lower, Counter())[journal] += 1
                if publish_time:
                    year_match = re.search(r'(\d{4})', publish_time)
                    if year_match:
                        author_years.setdefault(author_lower, []).append(int(year_match.group(1)))
        
        author_metrics = {}
        for author, paper_ids in author_papers.items():
            metrics = {
                'author_name': author,
                'paper_count': len(paper_ids),
                'papers': paper_ids,
                'unique_journals': len(author_journals[author]),
                'total_citations': 0,
                'avg_citations': 0.0,
                'h_index': 0,
                'recent_papers': 0,
            }
            
            if self.citation_builder:
                citation_counts = []
                for paper_id in paper_ids:
                    citations = self.citation_builder.get_citation_count(paper_id)
                    citation_counts.append(citations)
                    metrics['total_citations'] += citations
                
                if citation_counts:
                    metrics['avg_citations'] = np.mean(citation_counts)
                    metrics['h_index'] = self._calculate_h_index(citation_counts)
            
            if author_years[author]:
                metrics['recent_papers'] = sum(1 for year in author_years[author] if year >= 2020)
            
            author_metrics[author] = metrics
        
        self._calculate_prominence_scores(author_metrics)
        self.author_stats = author_metrics
        self.save_analysis()
    
    def _parse_authors(self, authors_str: str) -> List[str]:
        """Parse author names from metadata string."""
        authors = []
        if not authors_str or str(authors_str).lower() == 'nan':
            return authors
        
        authors_str = str(authors_str).strip()
        authors_str = re.sub(r'\([^)]*\)', '', authors_str)
        
        if ';' in authors_str:
            raw_authors = [a.strip() for a in authors_str.split(';') if a.strip()]
        else:
            raw_authors = [a.strip() for a in authors_str.split(',') if a.strip()]
        
        for raw_author in raw_authors:
            raw_author = re.sub(r'\b(PhD|MD|Dr\.?|Prof\.?)\b', '', raw_author, flags=re.IGNORECASE).strip()
            last_name = self._extract_last_name(raw_author)
            if last_name and len(last_name) > 1:
                authors.append(last_name)
        
        return authors
    
    def _extract_last_name(self, author_str: str) -> str:
        """Extract last name from author string."""
        if ',' in author_str:
            parts = author_str.split(',')
            if parts:
                return parts[0].strip().title()
        
        words = author_str.split()
        if words:
            last_word = words[-1]
            if len(last_word) == 1 and last_word.isupper():
                if len(words) > 1:
                    return words[-2].strip().title()
            return last_word.strip().title()
        
        return author_str.strip().title()
    
    def _calculate_h_index(self, citation_counts: List[int]) -> int:
        """Calculate h-index from citation counts."""
        if not citation_counts:
            return 0
        
        sorted_counts = sorted(citation_counts, reverse=True)
        h_index = 0
        for i, count in enumerate(sorted_counts, 1):
            if count >= i:
                h_index = i
            else:
                break
        
        return h_index
    
    def _calculate_prominence_scores(self, author_metrics: Dict):
        """Calculate normalized prominence score (0-1) for each author."""
        if not author_metrics:
            return
        
        paper_counts = [m['paper_count'] for m in author_metrics.values()]
        total_citations = [m['total_citations'] for m in author_metrics.values()]
        h_indices = [m['h_index'] for m in author_metrics.values()]
        recent_papers = [m['recent_papers'] for m in author_metrics.values()]
        
        max_papers = max(paper_counts) if paper_counts else 1
        max_citations = max(total_citations) if total_citations else 1
        max_h_index = max(h_indices) if h_indices else 1
        max_recent = max(recent_papers) if recent_papers else 1
        
        for author, metrics in author_metrics.items():
            paper_score = min(metrics['paper_count'] / max_papers, 1.0)
            citation_score = min(metrics['total_citations'] / max(max_citations, 1), 1.0)
            h_index_score = min(metrics['h_index'] / max(max_h_index, 1), 1.0)
            recent_score = min(metrics['recent_papers'] / max(max_recent, 1), 1.0)
            
            journal_bonus = min(metrics['unique_journals'] / 10.0, 0.2)
            
            total_score = (
                0.30 * paper_score +
                0.30 * citation_score +
                0.20 * h_index_score +
                0.20 * recent_score +
                journal_bonus
            )
            
            metrics['prominence_score'] = min(total_score, 1.0)
    
    def get_author_score(self, author_name: str) -> float:
        """Get prominence score for a specific author."""
        if not author_name:
            return 0.3
        
        author_key = author_name.lower().strip()
        
        if author_key in self.author_stats:
            metrics = self.author_stats[author_key]
            if metrics['paper_count'] >= 2 and metrics['prominence_score'] > 0.1:
                return metrics['prominence_score']
        
        for author, metrics in self.author_stats.items():
            if author_key in author or author in author_key:
                if metrics['paper_count'] >= 2 and metrics['prominence_score'] > 0.1:
                    return metrics['prominence_score']
        
        return 0.3
    
    def save_analysis(self, filepath: str = "author_prominence.pkl"):
        """Save author analysis to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'author_stats': self.author_stats
            }, f)
    
    def load_analysis(self, filepath: str = "author_prominence.pkl") -> bool:
        """Load author analysis from file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.author_stats = data.get('author_stats', {})
                    return True
            except:
                pass
        
        return False

# ============================================================================
# PAPER RANKER (Main Ranking System)
# ============================================================================

class PaperRanker:
    """Ranks papers based on multiple weighted criteria."""
    
    DEFAULT_WEIGHTS = {
        'citation_count': 0.30,
        'title_match': 0.25,
        'abstract_match': 0.15,
        'body_match': 0.10,
        'recency': 0.10,
        'journal_prestige': 0.05,
        'author_prominence': 0.05
    }
    
    PRESTIGIOUS_JOURNALS = {
        'lancet': 1.0, 'new england journal': 0.95, 'nature': 0.95,
        'science': 0.95, 'cell': 0.90, 'jama': 0.90, 'bmj': 0.85,
        'annals of internal medicine': 0.85, 'plos medicine': 0.80,
        'pnas': 0.80, 'nature medicine': 0.95, 'nejm': 0.95,
    }
    
    COVID_JOURNALS = {
        'journal of virology': 0.75, 'journal of infectious diseases': 0.75,
        'clinical infectious diseases': 0.75, 'emerging infectious diseases': 0.80,
        'eurosurveillance': 0.75, 'mmwr': 0.70,
    }
    
    def __init__(self, citation_builder, metadata_df: pd.DataFrame, 
                 weights: Dict[str, float] = None):
        self.citation_builder = citation_builder
        self.metadata_df = metadata_df
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}
        
        self.author_analyzer = AuthorProminenceAnalyzer(metadata_df, citation_builder)
        
        # Lazy initialization - only build when needed
        self.paper_metadata = None
        self.title_lookup = None
        self.journal_lookup = None
        self._metadata_initialized = False
    
    def _ensure_metadata_loaded(self):
        """Lazy load metadata on first use"""
        if self._metadata_initialized:
            return
        
        self.paper_metadata = {}
        self.title_lookup = {}
        self.journal_lookup = {}
        
        for _, row in self.metadata_df.iterrows():
            cord_uid = str(row['cord_uid']).strip()
            if cord_uid:
                self.paper_metadata[cord_uid] = row
                title = str(row.get('title', '')).lower()
                if title:
                    self.title_lookup[cord_uid] = title
                journal = str(row.get('journal', '')).lower()
                if journal:
                    self.journal_lookup[cord_uid] = journal
        
        self._metadata_initialized = True
    
    def calculate_paper_score(self, paper_id: str, query_terms: List[str], 
                            paper_text: Dict[str, str] = None) -> float:
        """Calculate comprehensive score for a single paper."""
        self._ensure_metadata_loaded()  # Ensure metadata is loaded
        
        if paper_id not in self.paper_metadata:
            return 0.0
        
        metadata = self.paper_metadata[paper_id]
        
        # Calculate individual component scores
        scores = {}
        scores['citation_count'] = self._calculate_citation_score(paper_id)
        
        if paper_text:
            scores.update(self._calculate_text_match_scores(paper_text, query_terms))
        else:
            title = str(metadata.get('title', '')).lower()
            abstract = str(metadata.get('abstract', '')).lower()
            scores['title_match'] = self._calculate_position_score(title, query_terms, 'title')
            scores['abstract_match'] = self._calculate_position_score(abstract, query_terms, 'abstract')
            scores['body_match'] = 0.0
        
        scores['recency'] = self._calculate_recency_score(metadata)
        scores['journal_prestige'] = self._calculate_journal_score(paper_id)
        scores['author_prominence'] = self._calculate_author_score(metadata)
        
        # Calculate weighted total
        total_score = 0.0
        for factor, weight in self.weights.items():
            if factor in scores:
                total_score += scores[factor] * weight
        
        return total_score
    
    def _calculate_citation_score(self, paper_id: str) -> float:
        """Calculate normalized citation score (0 to 1)."""
        if not self.citation_builder:
            return 0.3
        
        citation_count = self.citation_builder.get_citation_count(paper_id)
        if citation_count == 0:
            return 0.0
        
        log_citations = math.log1p(citation_count)
        max_log = math.log1p(100)
        score = min(log_citations / max_log, 1.0)
        
        if citation_count > 50:
            score = min(score * 1.1, 1.0)
        
        return score
    
    def _calculate_text_match_scores(self, paper_text: Dict[str, str], 
                                   query_terms: List[str]) -> Dict[str, float]:
        """Calculate scores based on where query terms appear."""
        scores = {}
        scores['title_match'] = self._calculate_position_score(
            paper_text.get('title', '').lower(), query_terms, 'title'
        )
        scores['abstract_match'] = self._calculate_position_score(
            paper_text.get('abstract', '').lower(), query_terms, 'abstract'
        )
        scores['body_match'] = self._calculate_position_score(
            paper_text.get('body', '').lower(), query_terms, 'body'
        )
        return scores
    
    def _calculate_position_score(self, text: str, query_terms: List[str], 
                                position: str) -> float:
        """Calculate score based on term frequency and position."""
        if not text or not query_terms:
            return 0.0
        
        position_params = {
            'title': {'weight': 1.0, 'max_terms': 5, 'exact_bonus': 0.5},
            'abstract': {'weight': 0.7, 'max_terms': 10, 'exact_bonus': 0.3},
            'body': {'weight': 0.3, 'max_terms': 20, 'exact_bonus': 0.1}
        }
        
        params = position_params.get(position, position_params['body'])
        
        term_scores = []
        found_terms = 0
        
        for term in query_terms:
            term_lower = term.lower()
            count = text.count(term_lower)
            
            if count == 0:
                continue
            
            found_terms += 1
            term_score = min(math.log1p(count) / math.log1p(params['max_terms']), 1.0)
            
            if len(query_terms) > 1:
                context_start = max(0, text.find(term_lower) - 50)
                context_end = min(len(text), text.find(term_lower) + len(term) + 50)
                context = text[context_start:context_end]
                other_terms = sum(1 for t in query_terms if t != term and t in context)
                if other_terms > 0:
                    term_score += (other_terms * 0.1)
            
            term_scores.append(term_score)
        
        if not term_scores:
            return 0.0
        
        avg_term_score = sum(term_scores) / len(term_scores)
        coverage = found_terms / len(query_terms)
        coverage_bonus = coverage * 0.2
        
        total_score = (avg_term_score + coverage_bonus)
        return min(total_score * params['weight'], 1.0)
    
    def _calculate_recency_score(self, metadata) -> float:
        """Calculate score based on publication date."""
        publish_time = str(metadata.get('publish_time', ''))
        
        if not publish_time or publish_time.lower() == 'nan':
            return 0.3
        
        try:
            year_match = re.search(r'(\d{4})', publish_time)
            if not year_match:
                return 0.3
            
            year = int(year_match.group(1))
            current_year = datetime.now().year
            
            if year >= 2020:
                if year == current_year or year == current_year - 1:
                    return 1.0
                elif year == 2022:
                    return 0.9
                elif year == 2021:
                    return 0.85
                elif year == 2020:
                    return 0.8
            elif year == 2019:
                return 0.6
            elif year >= 2015:
                return 0.4
            elif year >= 2010:
                return 0.3
            elif year >= 2000:
                return 0.2
            else:
                return 0.1
        except:
            return 0.3
    
    def _calculate_journal_score(self, paper_id: str) -> float:
        """Calculate score based on journal prestige."""
        self._ensure_metadata_loaded()  # Ensure metadata is loaded
        journal = self.journal_lookup.get(paper_id, '').lower()
        
        if not journal:
            return 0.3
        
        for journal_fragment, score in self.PRESTIGIOUS_JOURNALS.items():
            if journal_fragment in journal:
                return score
        
        for journal_fragment, score in self.COVID_JOURNALS.items():
            if journal_fragment in journal:
                return score
        
        quality_indicators = [
            ('journal', 0.5), ('proceedings', 0.4), ('letters', 0.4),
            ('transactions', 0.5), ('review', 0.6), ('international', 0.55),
        ]
        
        for indicator, score in quality_indicators:
            if indicator in journal:
                return score
        
        words = journal.split()
        if len(words) <= 3 and any(w.isalpha() for w in words):
            return 0.4
        
        return 0.35
    
    def _calculate_author_score(self, metadata) -> float:
        """Calculate score based on author prominence."""
        authors_str = str(metadata.get('authors', ''))
        
        if not authors_str or authors_str.lower() == 'nan':
            return 0.3
        
        parsed_authors = self._parse_authors_from_metadata(authors_str)
        if not parsed_authors:
            return 0.3
        
        author_scores = []
        for author in parsed_authors:
            score = self.author_analyzer.get_author_score(author)
            author_scores.append(score)
        
        if not author_scores:
            return 0.3
        
        max_score = max(author_scores)
        collaboration_bonus = 0.0
        author_count = len(parsed_authors)
        
        if author_count >= 10:
            collaboration_bonus = 0.15
        elif author_count >= 5:
            collaboration_bonus = 0.10
        elif author_count >= 3:
            collaboration_bonus = 0.05
        
        return min(max_score + collaboration_bonus, 1.0)
    
    def _parse_authors_from_metadata(self, authors_str: str) -> List[str]:
        """Extract author last names from metadata string."""
        return self.author_analyzer._parse_authors(authors_str)
    
    def rank_papers(self, paper_ids: List[str], query: str, 
                   paper_texts: Dict[str, Dict] = None) -> List[Tuple[str, float]]:
        """Rank a list of papers based on multiple criteria."""
        if not paper_ids:
            return []
        
        query_terms = self._preprocess_query(query)
        ranked_papers = []
        
        for paper_id in paper_ids:
            if paper_id not in self.paper_metadata:
                continue
            
            paper_text = paper_texts.get(paper_id) if paper_texts else None
            score = self.calculate_paper_score(paper_id, query_terms, paper_text)
            ranked_papers.append((paper_id, score))
        
        ranked_papers.sort(key=lambda x: x[1], reverse=True)
        return ranked_papers
    
    def _preprocess_query(self, query: str) -> List[str]:
        """Preprocess query terms."""
        if not query:
            return []
        
        query = query.lower().strip()
        words = re.findall(r'\b[a-z0-9]{3,}\b', query)
        
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
            'should', 'can', 'could', 'may', 'might', 'must', 'about', 'above',
            'after', 'before', 'between', 'from', 'into', 'through', 'during',
            'since', 'under', 'over', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
        }
        
        filtered_words = [w for w in words if w not in stopwords]
        medical_terms = {'sars', 'cov', 'covid', 'icu', 'rt', 'pcr', 'rna', 'dna', 'who'}
        short_medical = [w for w in words if w in medical_terms]
        
        result = filtered_words + short_medical
        seen = set()
        unique_result = []
        for w in result:
            if w not in seen:
                seen.add(w)
                unique_result.append(w)
        
        return unique_result

# ============================================================================
# INTEGRATION FUNCTIONS
# ============================================================================

def create_ranker(metadata_path: str, citation_graph_path: str = "citation_graph.pkl",
                 weights: Dict[str, float] = None) -> PaperRanker:
    """Factory function to create a PaperRanker instance."""
    from citation_graph import build_citation_graph
    
    metadata_df = pd.read_csv(metadata_path)
    citation_builder = build_citation_graph(metadata_path, citation_graph_path)
    ranker = PaperRanker(citation_builder, metadata_df, weights=weights)
    
    return ranker