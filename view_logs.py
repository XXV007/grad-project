"""
Log Viewer Utility
Provides easy access to application logs with filtering and analysis
"""

import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

LOG_FOLDER = os.path.join(os.path.dirname(__file__), 'logs')
MAIN_LOG = os.path.join(LOG_FOLDER, 'deepfake_detection.log')
ERROR_LOG = os.path.join(LOG_FOLDER, 'errors.log')

def print_separator(char='=', length=70):
    """Print a separator line"""
    print(char * length)

def tail_log(log_file, lines=50):
    """Show last N lines of a log file"""
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    print_separator()
    print(f"📄 Last {lines} lines of {os.path.basename(log_file)}")
    print_separator()
    
    with open(log_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line.rstrip())
    
    print_separator()

def show_errors_only(hours=24):
    """Show only error messages from the last N hours"""
    if not os.path.exists(ERROR_LOG):
        print("❌ Error log file not found")
        return
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    print_separator()
    print(f"🔴 Errors from the last {hours} hours")
    print_separator()
    
    error_count = 0
    with open(ERROR_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            # Try to parse timestamp (format: 2026-01-28 14:30:45)
            try:
                timestamp_str = line[:19]
                log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                if log_time >= cutoff_time:
                    print(line.rstrip())
                    error_count += 1
            except:
                # If timestamp parsing fails, include the line anyway
                print(line.rstrip())
                error_count += 1
    
    print_separator()
    print(f"Total errors found: {error_count}")

def search_logs(search_term):
    """Search for specific term in logs"""
    if not os.path.exists(MAIN_LOG):
        print("❌ Main log file not found")
        return
    
    print_separator()
    print(f"🔍 Search results for: '{search_term}'")
    print_separator()
    
    match_count = 0
    with open(MAIN_LOG, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if search_term.lower() in line.lower():
                print(f"Line {line_num}: {line.rstrip()}")
                match_count += 1
    
    print_separator()
    print(f"Total matches found: {match_count}")

def analyze_logs():
    """Analyze logs and show statistics"""
    if not os.path.exists(MAIN_LOG):
        print("❌ Main log file not found")
        return
    
    stats = {
        'DEBUG': 0,
        'INFO': 0,
        'WARNING': 0,
        'ERROR': 0,
        'CRITICAL': 0
    }
    
    module_stats = defaultdict(int)
    upload_count = 0
    analysis_count = 0
    
    with open(MAIN_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            # Count by level
            for level in stats.keys():
                if f' - {level} - ' in line:
                    stats[level] += 1
                    break
            
            # Count by module
            if ' - ' in line:
                parts = line.split(' - ')
                if len(parts) >= 2:
                    module = parts[1]
                    module_stats[module] += 1
            
            # Count operations
            if 'File uploaded' in line:
                upload_count += 1
            if 'Analysis complete' in line:
                analysis_count += 1
    
    print_separator()
    print("📊 Log Statistics")
    print_separator()
    
    print("\n📈 By Log Level:")
    for level, count in stats.items():
        bar = '█' * min(50, count)
        print(f"  {level:10s} : {count:5d} {bar}")
    
    print("\n📦 By Module (Top 10):")
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    for module, count in sorted_modules:
        bar = '█' * min(30, count // 5)
        print(f"  {module:20s} : {count:5d} {bar}")
    
    print("\n🎯 Operations:")
    print(f"  Videos Uploaded  : {upload_count}")
    print(f"  Videos Analyzed  : {analysis_count}")
    print(f"  Success Rate     : {(analysis_count/upload_count*100 if upload_count > 0 else 0):.1f}%")
    
    print_separator()

def show_menu():
    """Display interactive menu"""
    print_separator()
    print("🔍 Deepfake Detection System - Log Viewer")
    print_separator()
    print("\nOptions:")
    print("  1. Show last 50 lines of main log")
    print("  2. Show last 100 lines of main log")
    print("  3. Show all errors (last 24 hours)")
    print("  4. Show all errors (last 7 days)")
    print("  5. Search logs for specific term")
    print("  6. Analyze log statistics")
    print("  7. Show error log only")
    print("  0. Exit")
    print_separator()

def main():
    """Main interactive loop"""
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1].lower()
        
        if command == 'tail':
            lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            tail_log(MAIN_LOG, lines)
        
        elif command == 'errors':
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            show_errors_only(hours)
        
        elif command == 'search':
            if len(sys.argv) < 3:
                print("❌ Please provide search term")
                return
            search_logs(' '.join(sys.argv[2:]))
        
        elif command == 'stats':
            analyze_logs()
        
        else:
            print("❌ Unknown command")
            print("\nUsage:")
            print("  python view_logs.py tail [lines]       - Show last N lines")
            print("  python view_logs.py errors [hours]     - Show errors from last N hours")
            print("  python view_logs.py search <term>      - Search for term")
            print("  python view_logs.py stats              - Show statistics")
        
        return
    
    # Interactive mode
    while True:
        show_menu()
        choice = input("Enter your choice (0-7): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        
        elif choice == '1':
            tail_log(MAIN_LOG, 50)
        
        elif choice == '2':
            tail_log(MAIN_LOG, 100)
        
        elif choice == '3':
            show_errors_only(24)
        
        elif choice == '4':
            show_errors_only(24 * 7)
        
        elif choice == '5':
            search_term = input("Enter search term: ").strip()
            if search_term:
                search_logs(search_term)
        
        elif choice == '6':
            analyze_logs()
        
        elif choice == '7':
            tail_log(ERROR_LOG, 50)
        
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == '__main__':
    main()
