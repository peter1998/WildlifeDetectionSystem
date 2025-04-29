#!/usr/bin/env python3
# scripts/export_tool.py

import os
import sys
import json
import requests
import argparse
from pathlib import Path
from datetime import datetime
import yaml
from tabulate import tabulate

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ExportMonitor:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.export_dir = os.path.expanduser("~/Desktop/TU PHD/WildlifeDetectionSystem/data/export")
        
    def check_server_connection(self):
        """Check if the server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def get_datasets(self):
        """Retrieve available datasets from the API."""
        try:
            response = requests.get(f"{self.base_url}/api/annotations/datasets")
            if response.status_code != 200:
                print(f"Error fetching datasets: {response.text}")
                return []
            return response.json().get('datasets', [])
        except Exception as e:
            print(f"Error fetching datasets: {e}")
            return []
    
    def get_exports(self):
        """Retrieve existing exports from the API."""
        try:
            response = requests.get(f"{self.base_url}/api/annotations/exports")
            if response.status_code != 200:
                print(f"Error fetching exports: {response.text}")
                return []
            return response.json().get('exports', [])
        except Exception as e:
            print(f"Error fetching exports: {e}")
            return []
    
    def get_detailed_export_info(self, export_path, export_type):
        """Get detailed information about an export."""
        if not os.path.exists(export_path):
            return {"error": "Export directory not found"}
        
        info = {
            "path": export_path,
            "type": export_type,
            "size_mb": sum(f.stat().st_size for f in Path(export_path).glob('**/*') if f.is_file()) / (1024 * 1024),
            "created": datetime.fromtimestamp(os.path.getctime(export_path)).strftime('%Y-%m-%d %H:%M:%S'),
            "files": {"total": 0},
        }
        
        # Count files by type
        file_extensions = {}
        for f in Path(export_path).glob('**/*'):
            if f.is_file():
                info["files"]["total"] += 1
                ext = f.suffix.lower()
                if ext not in file_extensions:
                    file_extensions[ext] = 0
                file_extensions[ext] += 1
        
        info["files"]["by_extension"] = file_extensions
        
        # Get specific details based on export type
        if export_type == "YOLO":
            info.update(self._analyze_yolo_export(export_path))
        elif export_type == "COCO":
            info.update(self._analyze_coco_export(export_path))
        
        return info
    
    def _analyze_yolo_export(self, export_path):
        """Analyze a YOLO export directory."""
        yolo_info = {
            "train_images": 0,
            "val_images": 0,
            "train_labels": 0,
            "val_labels": 0,
            "classes": [],
            "class_distribution": {},
        }
        
        # Check for data.yaml
        data_yaml_path = os.path.join(export_path, 'data.yaml')
        if os.path.exists(data_yaml_path):
            try:
                with open(data_yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    yolo_info["yaml_config"] = yaml_data
            except Exception as e:
                yolo_info["yaml_error"] = str(e)
        
        # Check for classes.txt
        classes_path = os.path.join(export_path, 'classes.txt')
        if os.path.exists(classes_path):
            try:
                with open(classes_path, 'r') as f:
                    yolo_info["classes"] = [line.strip() for line in f.readlines()]
            except Exception as e:
                yolo_info["classes_error"] = str(e)
        
        # Count train/val images and labels
        train_images_dir = os.path.join(export_path, 'images', 'train')
        val_images_dir = os.path.join(export_path, 'images', 'val')
        train_labels_dir = os.path.join(export_path, 'labels', 'train')
        val_labels_dir = os.path.join(export_path, 'labels', 'val')
        
        if os.path.exists(train_images_dir):
            yolo_info["train_images"] = len([f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))])
        
        if os.path.exists(val_images_dir):
            yolo_info["val_images"] = len([f for f in os.listdir(val_images_dir) if os.path.isfile(os.path.join(val_images_dir, f))])
        
        if os.path.exists(train_labels_dir):
            yolo_info["train_labels"] = len([f for f in os.listdir(train_labels_dir) if os.path.isfile(os.path.join(train_labels_dir, f))])
        
        if os.path.exists(val_labels_dir):
            yolo_info["val_labels"] = len([f for f in os.listdir(val_labels_dir) if os.path.isfile(os.path.join(val_labels_dir, f))])
        
        # Check for export_report.json
        export_report_path = os.path.join(export_path, 'export_report.json')
        if os.path.exists(export_report_path):
            try:
                with open(export_report_path, 'r') as f:
                    export_report = json.load(f)
                    yolo_info["export_report"] = export_report
                    # Extract class distribution if available
                    if "species_distribution" in export_report:
                        yolo_info["class_distribution"] = export_report["species_distribution"]
            except Exception as e:
                yolo_info["report_error"] = str(e)
                
        # Analyze some labels to get class distribution if not available from report
        if not yolo_info["class_distribution"] and os.path.exists(train_labels_dir) and yolo_info["classes"]:
            class_counts = {i: 0 for i in range(len(yolo_info["classes"]))}
            
            # Sample a subset of labels for efficiency
            label_files = os.listdir(train_labels_dir)
            sample_size = min(500, len(label_files))
            sample_files = label_files[:sample_size]
            
            for label_file in sample_files:
                try:
                    with open(os.path.join(train_labels_dir, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:
                                class_id = int(parts[0])
                                if class_id in class_counts:
                                    class_counts[class_id] += 1
                except Exception:
                    pass
            
            # Create class distribution dictionary
            for class_id, count in class_counts.items():
                if count > 0 and class_id < len(yolo_info["classes"]):
                    class_name = yolo_info["classes"][class_id]
                    yolo_info["class_distribution"][class_name] = {"sampled_count": count}
        
        return yolo_info
    
    def _analyze_coco_export(self, export_path):
        """Analyze a COCO export directory."""
        coco_info = {
            "images_count": 0,
            "annotations_count": 0,
            "categories_count": 0,
            "categories": []
        }
        
        # Check for annotations.json
        coco_json_path = os.path.join(export_path, 'annotations.json')
        if os.path.exists(coco_json_path):
            try:
                with open(coco_json_path, 'r') as f:
                    coco_data = json.load(f)
                    
                    # Get counts
                    coco_info["images_count"] = len(coco_data.get("images", []))
                    coco_info["annotations_count"] = len(coco_data.get("annotations", []))
                    coco_info["categories_count"] = len(coco_data.get("categories", []))
                    
                    # Get categories
                    coco_info["categories"] = [cat.get("name") for cat in coco_data.get("categories", [])]
                    
                    # Count annotations per category
                    category_id_to_name = {cat.get("id"): cat.get("name") for cat in coco_data.get("categories", [])}
                    category_counts = {}
                    
                    for ann in coco_data.get("annotations", []):
                        cat_id = ann.get("category_id")
                        if cat_id in category_id_to_name:
                            cat_name = category_id_to_name[cat_id]
                            if cat_name not in category_counts:
                                category_counts[cat_name] = 0
                            category_counts[cat_name] += 1
                    
                    coco_info["category_distribution"] = category_counts
            except Exception as e:
                coco_info["json_error"] = str(e)
        
        # Count images
        images_dir = os.path.join(export_path, 'images')
        if os.path.exists(images_dir):
            coco_info["image_files"] = len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
        
        return coco_info
    
    def print_dataset_summary(self, datasets):
        """Print summary of available datasets."""
        if not datasets:
            print("No datasets found.")
            return
        
        headers = ["Name", "Total", "Annotated", "Completion %", "Species"]
        rows = []
        
        for dataset in datasets:
            rows.append([
                dataset['name'],
                dataset['total_images'],
                dataset['annotated_images'],
                f"{dataset['completion_percentage']:.1f}%",
                dataset['species_count']
            ])
        
        print("\n=== Available Datasets ===")
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    
    def print_exports_summary(self, exports):
        """Print summary of existing exports."""
        if not exports:
            print("No exports found.")
            return
        
        headers = ["Name", "Type", "Created", "Size"]
        rows = []
        
        for export in exports:
            size_mb = export['size_bytes'] / (1024 * 1024)
            created = export['created'].split('T')[0] + ' ' + export['created'].split('T')[1][:8]
            rows.append([
                export['name'],
                export['type'],
                created,
                f"{size_mb:.2f} MB"
            ])
        
        print("\n=== Existing Exports ===")
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    
    def print_export_details(self, export_info):
        """Print detailed information about an export."""
        if "error" in export_info:
            print(f"Error: {export_info['error']}")
            return
        
        print(f"\n=== Export Details: {os.path.basename(export_info['path'])} ===")
        print(f"Type: {export_info['type']}")
        print(f"Created: {export_info['created']}")
        print(f"Size: {export_info['size_mb']:.2f} MB")
        print(f"Total Files: {export_info['files']['total']}")
        
        # Print file distribution
        if export_info['files']['by_extension']:
            print("\nFile Types:")
            for ext, count in sorted(export_info['files']['by_extension'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {ext}: {count}")
        
        # Print YOLO-specific info
        if export_info['type'] == "YOLO":
            print("\nYOLO Dataset Structure:")
            print(f"  Training Images: {export_info['train_images']}")
            print(f"  Validation Images: {export_info['val_images']}")
            print(f"  Training Labels: {export_info['train_labels']}")
            print(f"  Validation Labels: {export_info['val_labels']}")
            
            # Print class info
            if export_info['classes']:
                print(f"\nClasses ({len(export_info['classes'])}):")
                for i, cls in enumerate(export_info['classes']):
                    print(f"  {i}: {cls}")
            
            # Print distribution info from report
            if "export_report" in export_info and "species_distribution" in export_info["export_report"]:
                print("\nClass Distribution:")
                rows = []
                headers = ["Class", "Train", "Val", "Total", "Train %", "Val %"]
                
                for class_name, data in export_info["export_report"]["species_distribution"].items():
                    train_count = data.get("train_count", 0)
                    val_count = data.get("val_count", 0)
                    total_count = data.get("total_count", train_count + val_count)
                    train_pct = data.get("train_percentage", 0)
                    val_pct = data.get("val_percentage", 0)
                    
                    rows.append([
                        class_name, 
                        train_count, 
                        val_count, 
                        total_count,
                        f"{train_pct:.1f}%", 
                        f"{val_pct:.1f}%"
                    ])
                
                print(tabulate(rows, headers=headers, tablefmt="simple"))
            
            # Print dataset summary from report
            if "export_report" in export_info and "dataset_summary" in export_info["export_report"]:
                summary = export_info["export_report"]["dataset_summary"]
                print("\nDataset Summary:")
                for key, value in summary.items():
                    if key != "export_timestamp":
                        print(f"  {key}: {value}")
        
        # Print COCO-specific info
        elif export_info['type'] == "COCO":
            print("\nCOCO Dataset Structure:")
            print(f"  Images: {export_info['images_count']}")
            print(f"  Annotations: {export_info['annotations_count']}")
            print(f"  Categories: {export_info['categories_count']}")
            
            # Print category distribution
            if "category_distribution" in export_info:
                print("\nCategory Distribution:")
                rows = []
                headers = ["Category", "Annotations", "Percentage"]
                
                total = sum(export_info["category_distribution"].values())
                for cat, count in sorted(export_info["category_distribution"].items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total) * 100 if total > 0 else 0
                    rows.append([cat, count, f"{percentage:.1f}%"])
                
                print(tabulate(rows, headers=headers, tablefmt="simple"))
    
    def run(self, args):
        """Run the export monitor with specified arguments."""
        if not self.check_server_connection():
            print("Could not connect to server. Make sure it's running on http://localhost:5000")
            return
        
        # Get datasets and exports info
        datasets = self.get_datasets()
        exports = self.get_exports()
        
        # Print list of datasets and exports if requested
        if args.list or (not args.analyze):
            self.print_dataset_summary(datasets)
            self.print_exports_summary(exports)
        
        # Analyze a specific export
        if args.analyze:
            found = False
            for export in exports:
                if export['name'] == args.analyze:
                    export_info = self.get_detailed_export_info(export['path'], export['type'])
                    self.print_export_details(export_info)
                    found = True
                    break
            
            if not found:
                print(f"Export '{args.analyze}' not found. Use --list to see available exports.")

def main():
    parser = argparse.ArgumentParser(description='Wildlife Detection System Export Monitor')
    parser.add_argument('--list', action='store_true', help='List available datasets and existing exports')
    parser.add_argument('--analyze', type=str, help='Analyze a specific export in detail')
    
    args = parser.parse_args()
    
    print("\n=== Wildlife Detection System Export Monitor ===")
    
    monitor = ExportMonitor()
    monitor.run(args)
    
    print("\n=== Monitor Completed ===")

if __name__ == "__main__":
    main()