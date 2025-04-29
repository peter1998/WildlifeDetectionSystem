#!/usr/bin/env python3
# scripts/export_report.py

import os
import sys
import json
import argparse
import traceback
from datetime import datetime
import subprocess
from fpdf import FPDF
import yaml
import logging

# Configure enhanced logging
logging.basicConfig(
    level=logging.DEBUG,  # Switch to DEBUG for more detailed messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to stdout for capture
    ]
)
logger = logging.getLogger("export_report")

# Print environment information for debugging
logger.debug(f"Python version: {sys.version}")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Script path: {os.path.abspath(__file__)}")

class ExportReportGenerator:
    def __init__(self):
        # More robust path detection
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.debug(f"Script directory: {self.script_dir}")
        
        # Check for API scripts directory
        if os.path.basename(self.script_dir) == 'scripts' and os.path.basename(os.path.dirname(self.script_dir)) == 'api':
            self.project_root = os.path.dirname(os.path.dirname(self.script_dir))
        else:
            self.project_root = os.path.dirname(self.script_dir)
        
        logger.debug(f"Project root: {self.project_root}")
        
        # Define export directory path
        self.export_dir = os.path.join(self.project_root, "data/export")
        logger.debug(f"Export directory: {self.export_dir}")
        if not os.path.exists(self.export_dir):
            logger.warning(f"Export directory does not exist: {self.export_dir}")
            # Try to find the export directory
            data_dir = os.path.join(self.project_root, "data")
            if os.path.exists(data_dir):
                for item in os.listdir(data_dir):
                    if item.lower() == "export" or "export" in item.lower():
                        self.export_dir = os.path.join(data_dir, item)
                        logger.debug(f"Found alternative export dir: {self.export_dir}")
                        break
        
        # Define reports directory path
        self.reports_dir = os.path.join(self.project_root, "reports")
        logger.debug(f"Reports directory: {self.reports_dir}")
        
        # Create reports directory if it doesn't exist
        try:
            os.makedirs(self.reports_dir, exist_ok=True)
            logger.debug(f"Created reports directory: {self.reports_dir}")
        except Exception as e:
            logger.error(f"Error creating reports directory: {e}")
            # Try to use a fallback directory
            self.reports_dir = os.path.join(self.script_dir, "reports")
            logger.debug(f"Using fallback reports directory: {self.reports_dir}")
            os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_report(self, export_name):
        """Generate PDF and text reports for the export."""
        try:
            logger.debug(f"Starting report generation for: {export_name}")
            
            # Check if export exists
            export_path = os.path.join(self.export_dir, export_name)
            logger.debug(f"Checking export path: {export_path}")
            
            if not os.path.exists(export_path):
                logger.error(f"Export not found: {export_path}")
                # Try listing available exports
                if os.path.exists(self.export_dir):
                    available_exports = os.listdir(self.export_dir)
                    logger.debug(f"Available exports: {available_exports}")
                return None
            
            # Determine export type
            export_type = "UNKNOWN"
            if export_name.startswith("yolo_"):
                export_type = "YOLO"
            elif export_name.startswith("coco_"):
                export_type = "COCO"
            
            logger.debug(f"Export type: {export_type}")
            
            # Generate text report directly
            logger.debug(f"Generating text report...")
            txt_report = self.generate_text_report_direct(export_name, export_type, export_path)
            
            # Generate PDF report
            logger.debug(f"Generating PDF report...")
            pdf_report = self.generate_pdf_report(export_name, export_type, txt_report, export_path)
            
            if not txt_report and not pdf_report:
                logger.error("Both text and PDF report generation failed")
                return None
            
            return {
                "text_report": txt_report,
                "pdf_report": pdf_report
            }
        except Exception as e:
            logger.error(f"Error in generate_report: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_text_report_direct(self, export_name, export_type, export_path):
        """Generate a text report directly without using export_tool."""
        try:
            # Define output file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.reports_dir, f"report_{export_name}_{timestamp}.txt")
            logger.debug(f"Text report will be saved to: {output_file}")
            
            # Generate report content
            with open(output_file, 'w') as f:
                f.write(f"=== Wildlife Detection System Export Report ===\n\n")
                f.write(f"Export Name: {export_name}\n")
                f.write(f"Export Type: {export_type}\n")
                f.write(f"Export Path: {export_path}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Calculate total size
                total_size = 0
                file_count = 0
                file_extensions = {}
                
                logger.debug(f"Scanning export directory for files: {export_path}")
                for root, dirs, files in os.walk(export_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        file_count += 1
                        
                        # Track file extensions
                        ext = os.path.splitext(file)[1].lower()
                        if ext not in file_extensions:
                            file_extensions[ext] = 0
                        file_extensions[ext] += 1
                
                # Write size info
                f.write(f"Total Files: {file_count}\n")
                f.write(f"Total Size: {total_size / (1024*1024):.2f} MB\n\n")
                
                # Write file type distribution
                f.write("File Types:\n")
                for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {ext}: {count}\n")
                f.write("\n")
                
                # Add export-specific information
                if export_type == "YOLO":
                    logger.debug("Writing YOLO-specific info")
                    self._write_yolo_info(f, export_path)
                elif export_type == "COCO":
                    logger.debug("Writing COCO-specific info")
                    self._write_coco_info(f, export_path)
            
            logger.info(f"Text report generated: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error generating text report: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _write_yolo_info(self, file, export_path):
        """Write YOLO-specific information to the text report."""
        try:
            file.write("YOLO Dataset Structure:\n")
            
            # Count train/val images and labels
            train_images_dir = os.path.join(export_path, 'images', 'train')
            val_images_dir = os.path.join(export_path, 'images', 'val')
            train_labels_dir = os.path.join(export_path, 'labels', 'train')
            val_labels_dir = os.path.join(export_path, 'labels', 'val')
            
            logger.debug(f"Checking YOLO directories: {train_images_dir}, {val_images_dir}, {train_labels_dir}, {val_labels_dir}")
            
            train_images = 0
            val_images = 0
            train_labels = 0
            val_labels = 0
            
            if os.path.exists(train_images_dir):
                train_images = len([f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))])
            else:
                logger.warning(f"Train images directory not found: {train_images_dir}")
            
            if os.path.exists(val_images_dir):
                val_images = len([f for f in os.listdir(val_images_dir) if os.path.isfile(os.path.join(val_images_dir, f))])
            else:
                logger.warning(f"Validation images directory not found: {val_images_dir}")
            
            if os.path.exists(train_labels_dir):
                train_labels = len([f for f in os.listdir(train_labels_dir) if os.path.isfile(os.path.join(train_labels_dir, f))])
            else:
                logger.warning(f"Train labels directory not found: {train_labels_dir}")
            
            if os.path.exists(val_labels_dir):
                val_labels = len([f for f in os.listdir(val_labels_dir) if os.path.isfile(os.path.join(val_labels_dir, f))])
            else:
                logger.warning(f"Validation labels directory not found: {val_labels_dir}")
            
            file.write(f"  Training Images: {train_images}\n")
            file.write(f"  Validation Images: {val_images}\n")
            file.write(f"  Training Labels: {train_labels}\n")
            file.write(f"  Validation Labels: {val_labels}\n\n")
            
            # Read classes.txt
            classes_path = os.path.join(export_path, 'classes.txt')
            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                
                file.write(f"Classes ({len(classes)}):\n")
                for i, cls in enumerate(classes):
                    file.write(f"  {i}: {cls}\n")
                file.write("\n")
            else:
                logger.warning(f"Classes file not found: {classes_path}")
            
            # Read export_report.json
            report_path = os.path.join(export_path, 'export_report.json')
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    try:
                        report_data = json.load(f)
                        
                        if 'dataset_summary' in report_data:
                            file.write("Dataset Summary:\n")
                            for key, value in report_data['dataset_summary'].items():
                                if key != 'export_timestamp':
                                    file.write(f"  {key}: {value}\n")
                            file.write("\n")
                        
                        if 'species_distribution' in report_data:
                            file.write("Class Distribution:\n")
                            file.write(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Total':<8} {'Train %':<10} {'Val %':<10}\n")
                            file.write("-" * 70 + "\n")
                            
                            for class_name, data in report_data['species_distribution'].items():
                                train_count = data.get('train_count', 0)
                                val_count = data.get('val_count', 0)
                                total_count = data.get('total_count', train_count + val_count)
                                train_pct = data.get('train_percentage', 0)
                                val_pct = data.get('val_percentage', 0)
                                
                                file.write(f"{class_name:<20} {train_count:<8} {val_count:<8} {total_count:<8} {train_pct:.1f}%{'':<5} {val_pct:.1f}%{'':<5}\n")
                            
                            file.write("\n")
                    except Exception as e:
                        logger.error(f"Error reading export report: {e}")
                        file.write(f"Error reading export report: {e}\n\n")
            else:
                logger.warning(f"Export report not found: {report_path}")
        except Exception as e:
            logger.error(f"Error in _write_yolo_info: {e}")
            logger.error(traceback.format_exc())
            file.write(f"Error generating YOLO info: {e}\n\n")
    
    def _write_coco_info(self, file, export_path):
        """Write COCO-specific information to the text report."""
        try:
            file.write("COCO Dataset Structure:\n")
            
            # Read annotations.json
            json_path = os.path.join(export_path, 'annotations.json')
            logger.debug(f"Looking for COCO annotations at: {json_path}")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        coco_data = json.load(f)
                    
                    images_count = len(coco_data.get('images', []))
                    annotations_count = len(coco_data.get('annotations', []))
                    categories_count = len(coco_data.get('categories', []))
                    
                    file.write(f"  Images: {images_count}\n")
                    file.write(f"  Annotations: {annotations_count}\n")
                    file.write(f"  Categories: {categories_count}\n\n")
                    
                    # Write categories
                    if 'categories' in coco_data:
                        file.write(f"Categories ({categories_count}):\n")
                        for cat in coco_data['categories']:
                            file.write(f"  {cat.get('id')}: {cat.get('name')}\n")
                        file.write("\n")
                    
                    # Calculate category distribution
                    if 'annotations' in coco_data and 'categories' in coco_data:
                        category_id_to_name = {cat.get('id'): cat.get('name') for cat in coco_data.get('categories', [])}
                        category_counts = {}
                        
                        for ann in coco_data.get('annotations', []):
                            cat_id = ann.get('category_id')
                            if cat_id in category_id_to_name:
                                cat_name = category_id_to_name[cat_id]
                                if cat_name not in category_counts:
                                    category_counts[cat_name] = 0
                                category_counts[cat_name] += 1
                        
                        if category_counts:
                            file.write("Category Distribution:\n")
                            file.write(f"{'Category':<20} {'Count':<8} {'Percentage':<12}\n")
                            file.write("-" * 40 + "\n")
                            
                            total = sum(category_counts.values())
                            for cat_name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / total) * 100 if total > 0 else 0
                                file.write(f"{cat_name:<20} {count:<8} {percentage:.1f}%{'':<7}\n")
                            
                            file.write("\n")
                except Exception as e:
                    logger.error(f"Error reading COCO annotations: {e}")
                    file.write(f"Error reading COCO annotations: {e}\n\n")
            else:
                logger.warning(f"COCO annotations file not found: {json_path}")
                file.write(f"  COCO annotations file not found\n\n")
        except Exception as e:
            logger.error(f"Error in _write_coco_info: {e}")
            logger.error(traceback.format_exc())
            file.write(f"Error generating COCO info: {e}\n\n")
    
    def generate_pdf_report(self, export_name, export_type, txt_report, export_path=None):
        """Generate a PDF report from the export data and text report."""
        try:
            # Define output file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.reports_dir, f"report_{export_name}_{timestamp}.pdf")
            logger.debug(f"PDF report will be saved to: {output_file}")
            
            if not export_path:
                export_path = os.path.join(self.export_dir, export_name)
            
            # Initialize PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Add title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"Wildlife Detection System - {export_type} Export Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Export: {export_name}", ln=True, align="C")
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            
            # Add export details
            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Export Details", ln=True)
            pdf.set_font("Arial", "", 12)
            
            # Key metrics section
            if export_type == "YOLO":
                # Read data.yaml for YOLO exports
                yaml_path = os.path.join(export_path, "data.yaml")
                logger.debug(f"Looking for YOLO data.yaml at: {yaml_path}")
                
                if os.path.exists(yaml_path):
                    try:
                        with open(yaml_path, 'r') as f:
                            yaml_data = yaml.safe_load(f)
                            
                        pdf.cell(0, 10, f"Training Images: {os.path.basename(yaml_data.get('train', 'Unknown'))}", ln=True)
                        pdf.cell(0, 10, f"Validation Images: {os.path.basename(yaml_data.get('val', 'Unknown'))}", ln=True)
                        pdf.cell(0, 10, f"Number of Classes: {yaml_data.get('nc', 'Unknown')}", ln=True)
                    except Exception as e:
                        logger.error(f"Error reading data.yaml: {e}")
                        pdf.cell(0, 10, f"Error reading data.yaml: {str(e)}", ln=True)
                else:
                    logger.warning(f"YOLO data.yaml not found: {yaml_path}")
                    pdf.cell(0, 10, "Data.yaml file not found", ln=True)
                        
                # Read export_report.json for more detailed stats
                report_path = os.path.join(export_path, "export_report.json")
                logger.debug(f"Looking for export_report.json at: {report_path}")
                
                if os.path.exists(report_path):
                    try:
                        with open(report_path, 'r') as f:
                            report_data = json.load(f)
                        
                        if 'dataset_summary' in report_data:
                            summary = report_data['dataset_summary']
                            
                            pdf.ln(5)
                            pdf.set_font("Arial", "B", 14)
                            pdf.cell(0, 10, "Dataset Summary", ln=True)
                            pdf.set_font("Arial", "", 12)
                            
                            for key, value in summary.items():
                                if key != 'export_timestamp':
                                    pdf.cell(0, 10, f"{key.replace('_', ' ').title()}: {value}", ln=True)
                    except Exception as e:
                        logger.error(f"Error reading export_report.json: {e}")
                        pdf.cell(0, 10, f"Error reading export report: {str(e)}", ln=True)
                else:
                    logger.warning(f"Export report not found: {report_path}")
                
            elif export_type == "COCO":
                # For COCO exports
                json_path = os.path.join(export_path, "annotations.json")
                logger.debug(f"Looking for COCO annotations.json at: {json_path}")
                
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        
                        pdf.cell(0, 10, f"Images: {len(json_data.get('images', []))}", ln=True)
                        pdf.cell(0, 10, f"Annotations: {len(json_data.get('annotations', []))}", ln=True)
                        pdf.cell(0, 10, f"Categories: {len(json_data.get('categories', []))}", ln=True)
                    except Exception as e:
                        logger.error(f"Error reading annotations.json: {e}")
                        pdf.cell(0, 10, f"Error reading annotations.json: {str(e)}", ln=True)
                else:
                    logger.warning(f"COCO annotations file not found: {json_path}")
                    pdf.cell(0, 10, "Annotations.json file not found", ln=True)
            
            # If text report exists, add its content
            if txt_report and os.path.exists(txt_report):
                logger.debug(f"Adding text report content from: {txt_report}")
                
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Detailed Analysis", ln=True)
                pdf.set_font("Courier", "", 10)
                
                try:
                    with open(txt_report, 'r') as f:
                        # Read the content by lines to handle line breaks
                        for line in f:
                            # Remove ANSI color codes if present
                            line = line.strip()
                            if line:
                                # Handle lines that are too long
                                while len(line) > 90:
                                    pdf.cell(0, 6, line[:90], ln=True)
                                    line = line[90:]
                                pdf.cell(0, 6, line, ln=True)
                except Exception as e:
                    logger.error(f"Error adding text report content: {e}")
                    pdf.cell(0, 10, f"Error including text report: {str(e)}", ln=True)
            else:
                logger.warning(f"Text report not available or not readable: {txt_report}")
            
            # Save PDF
            pdf.output(output_file)
            logger.info(f"PDF report generated: {output_file}")
            
            return output_file
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            logger.error(traceback.format_exc())
            return None

def main():
    parser = argparse.ArgumentParser(description="Generate reports for an export")
    parser.add_argument("export_name", help="Name of the export directory")
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting report generation for: {args.export_name}")
        
        generator = ExportReportGenerator()
        reports = generator.generate_report(args.export_name)
        
        if reports:
            print(f"Reports generated successfully!")
            print(f"Text report: {reports['text_report']}")
            print(f"PDF report: {reports['pdf_report']}")
        else:
            print("Failed to generate reports.")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        print("Failed to generate reports.")

if __name__ == "__main__":
    main()