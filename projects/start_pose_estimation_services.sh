#!/bin/bash

# AI Multi-Person Pose Estimation System Startup Script
# This script will start independent YOLOX, RTMPose, ByteTrack and Annotation Docker services

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${BLUE}[====== $1 ======]${NC}"
}

# Check if configuration file exists
check_config() {
    if [ ! -f "./pose_estimation_config.yaml" ]; then
        log_error "Configuration file not found: ./pose_estimation_config.yaml"
        log_info "Please ensure the pose_estimation_config.yaml file exists in the project root directory"
        exit 1
    fi
    log_info "Configuration file found: ./pose_estimation_config.yaml"
}

# Check if Docker and docker-compose are available
check_dependencies() {
    log_header "Checking Dependencies"
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed or not in PATH"
        exit 1
    fi
    
    log_info "Docker and docker-compose are available"
}

# Stop existing services
stop_services() {
    log_header "Stopping Existing Services"
    docker-compose down || true
    log_info "All services stopped"
}

# Build and start base services
start_base_services() {
    log_header "Starting Base Services"
    log_info "Starting MQTT broker and RTSP server..."
    docker-compose up -d mqtt-broker rtsp-server
    
    log_info "Waiting for base services to be ready..."
    sleep 1
    
    log_info "Starting video source services..."
    docker-compose up -d mp4-rtsp-source mp4-transcode-sei
    
    log_info "Waiting for video services to be ready..."
    sleep 1
}

# Start AI services (in dependency order)
start_ai_services() {
    log_header "Starting AI Processing Services"
    
    # 1. Start YOLOX service
    log_info "Starting YOLOX detection service..."
    docker-compose up -d yolox-service
    log_info "Waiting for YOLOX to initialize..."
    sleep 1
    
    # 2. Start RTMPose and ByteTrack services (can be started in parallel since both depend on YOLOX)
    log_info "Starting RTMPose and ByteTrack services..."
    docker-compose up -d rtmpose-service bytetrack-service
    log_info "Waiting for RTMPose and ByteTrack to initialize..."
    sleep 1
    
    # 3. Start Annotation service (depends on RTMPose and ByteTrack)
    log_info "Starting Annotation visualization service..."
    docker-compose up -d annotation-service
    log_info "Waiting for Annotation service to initialize..."
    sleep 1
}

# Check service status
check_services() {
    log_header "Checking Service Status"
    
    services=("mqtt-broker" "rtsp-server" "mp4_rtsp_container" "mp4-transcoder-sei_container" 
              "yolox-service" "rtmpose-service" "bytetrack-service" "annotation-service")
    
    for service in "${services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            log_info "$service: ✓ Running"
        else
            log_error "$service: ✗ Not running"
        fi
    done
}

# Show service URLs
show_urls() {
    log_header "Service Access Information"
    echo ""
    echo "📺 RTSP Stream URLs:"
    echo "   Input stream (raw video): rtsp://localhost:8554/rawstream"
    echo "   Processed stream: rtsp://localhost:8554/mystream"
    echo "   Output stream (annotated): rtsp://localhost:8554/outstream"
    echo ""
    echo "📡 MQTT Broker:"
    echo "   Address: localhost:1883"
    echo "   Topics:"
    echo "     - yolox: YOLOX detection results"
    echo "     - rtmpose: RTMPose pose estimation results"
    echo "     - bytetrack: ByteTrack tracking results"
    echo ""
    echo "🔧 Docker Management Commands:"
    echo "   View logs: docker-compose logs -f [service-name]"
    echo "   Stop services: docker-compose down"
    echo "   Restart service: docker-compose restart [service-name]"
    echo ""
}

# Show help information
show_help() {
    echo "AI Multi-Person Pose Estimation System Startup Script"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  start     Start all services (default)"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  status    View service status"
    echo "  logs      View service logs"
    echo "  help      Show this help information"
    echo ""
}

# Main function
main() {
    local action="${1:-start}"
    
    case "$action" in
        "start")
            log_header "AI Multi-Person Pose Estimation System Startup"
            check_dependencies
            check_config
            stop_services
            start_base_services
            start_ai_services
            check_services
            show_urls
            log_info "All services started successfully!"
            ;;
        "start_base_services")
            log_header "Starting Base Services Only"
            stop_services
            start_base_services
            ;;
        "stop")
            log_header "Stopping All Services"
            stop_services
            log_info "All services stopped successfully!"
            ;;
        "restart")
            log_header "Restarting All Services"
            stop_services
            sleep 5
            main "start"
            ;;
        "status")
            check_services
            ;;
        "logs")
            log_info "Showing logs for all services..."
            docker-compose logs -f
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown action: $action"
            show_help
            exit 1
            ;;

    esac
}

# Execute main function with all arguments
main "$@" 