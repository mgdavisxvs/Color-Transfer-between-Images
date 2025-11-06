"""
OpenAPI 3.0 Specification for Flask Color Transfer API

This specification provides comprehensive documentation for all API endpoints
with request/response schemas, examples, and error responses.
"""

openapi_spec = {
    "openapi": "3.0.0",
    "info": {
        "title": "RAL Color Transfer API",
        "description": """
# RAL Color Transfer API

A professional color transfer API using the RAL color palette with Delta E (ΔE) color matching.

## Features

- **188 RAL Colors**: Professional-grade color palette
- **Delta E Calculations**: CIEDE2000 perceptual color difference
- **Color Transfer Methods**: Reinhard statistical transfer, auto-matching
- **Quality Control**: ΔE statistics, thresholds, and acceptance criteria
- **Async Processing**: Background task execution for long operations
- **Security**: CSRF protection, rate limiting, input validation

## Authentication

Currently no authentication required. Rate limits apply:
- Global: 200 requests/day, 50/hour
- Upload: 10/minute
- Processing: 5/minute

## Error Responses

All errors follow this format:
```json
{
  "success": false,
  "error": "Error message",
  "error_type": "error_category"
}
```

Error types:
- `validation_error` (400)
- `not_found` (404)
- `file_too_large` (413)
- `rate_limit` (429)
- `internal_error` (500)

## Rate Limiting

Rate limit headers included in responses:
- `X-RateLimit-Limit`: Total requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when limit resets
        """,
        "version": "2.0.0",
        "contact": {
            "name": "API Support",
            "email": "support@example.com"
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    },
    "servers": [
        {
            "url": "http://localhost:5000",
            "description": "Development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ],
    "tags": [
        {
            "name": "palette",
            "description": "RAL color palette operations"
        },
        {
            "name": "color-matching",
            "description": "Color matching and Delta E calculations"
        },
        {
            "name": "image-processing",
            "description": "Image upload and color transfer"
        },
        {
            "name": "async-tasks",
            "description": "Asynchronous task management"
        },
        {
            "name": "system",
            "description": "System health and information"
        }
    ],
    "paths": {
        "/api/palette": {
            "get": {
                "tags": ["palette"],
                "summary": "Get RAL color palette",
                "description": "Retrieve all RAL colors or search by name",
                "parameters": [
                    {
                        "name": "search",
                        "in": "query",
                        "description": "Search query for color names (case-insensitive)",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "example": "red"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean", "example": True},
                                        "total": {"type": "integer", "example": 188},
                                        "colors": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/RALColor"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/palette/stats": {
            "get": {
                "tags": ["palette"],
                "summary": "Get palette statistics",
                "description": "Get statistical information about the RAL color palette",
                "responses": {
                    "200": {
                        "description": "Palette statistics",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "statistics": {
                                            "$ref": "#/components/schemas/PaletteStats"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/color/match": {
            "post": {
                "tags": ["color-matching"],
                "summary": "Find closest RAL color matches",
                "description": "Find the closest matching RAL colors for a given RGB color",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/ColorMatchRequest"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Matching colors found",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ColorMatchResponse"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Validation error",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/upload": {
            "post": {
                "tags": ["image-processing"],
                "summary": "Upload image file",
                "description": "Upload a source image for color transfer processing",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "format": "binary",
                                        "description": "Image file (PNG, JPG, BMP, TIFF)"
                                    }
                                },
                                "required": ["file"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Upload successful",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/UploadResponse"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid file",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "413": {
                        "description": "File too large (max 50MB)",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/process/reinhard": {
            "post": {
                "tags": ["image-processing"],
                "summary": "Process color transfer (Reinhard method)",
                "description": "Apply Reinhard statistical color transfer to match a RAL color",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/ProcessRequest"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Processing complete",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ProcessResponse"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Source image not found",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/process/async": {
            "post": {
                "tags": ["async-tasks", "image-processing"],
                "summary": "Process color transfer asynchronously",
                "description": "Submit color transfer job for background processing",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/ProcessRequest"
                            }
                        }
                    }
                },
                "responses": {
                    "202": {
                        "description": "Task accepted for processing",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/AsyncTaskResponse"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/task/{task_id}": {
            "get": {
                "tags": ["async-tasks"],
                "summary": "Get task status",
                "description": "Check the status of an asynchronous task",
                "parameters": [
                    {
                        "name": "task_id",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Task ID returned from async endpoint"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Task status",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/TaskStatusResponse"
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Task not found",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Error"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/delta-e/compute": {
            "post": {
                "tags": ["color-matching"],
                "summary": "Compute Delta E between two colors",
                "description": "Calculate the perceptual color difference (CIEDE2000)",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/DeltaERequest"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Delta E calculated",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/DeltaEResponse"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/health": {
            "get": {
                "tags": ["system"],
                "summary": "Health check",
                "description": "Check if the API is running and healthy",
                "responses": {
                    "200": {
                        "description": "Service is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/HealthResponse"
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "RALColor": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "example": "RAL 3000"},
                    "name": {"type": "string", "example": "Flame red"},
                    "rgb": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "example": [171, 32, 24]
                    },
                    "hex": {"type": "string", "example": "#AB2018"}
                }
            },
            "PaletteStats": {
                "type": "object",
                "properties": {
                    "total_colors": {"type": "integer"},
                    "rgb_range": {"type": "object"},
                    "lab_range": {"type": "object"}
                }
            },
            "ColorMatchRequest": {
                "type": "object",
                "required": ["rgb"],
                "properties": {
                    "rgb": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 255},
                        "minItems": 3,
                        "maxItems": 3,
                        "example": [255, 0, 0]
                    },
                    "top_n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "example": 3
                    }
                }
            },
            "ColorMatchResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "input_rgb": {"type": "array", "items": {"type": "integer"}},
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "color": {"$ref": "#/components/schemas/RALColor"},
                                "delta_e": {"type": "number", "example": 5.23},
                                "interpretation": {"type": "string", "example": "Clear difference"}
                            }
                        }
                    }
                }
            },
            "UploadResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "job_id": {"type": "string", "example": "20251106_120000_a1b2c3d4"},
                    "filename": {"type": "string"},
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "width": {"type": "integer"},
                            "height": {"type": "integer"}
                        }
                    },
                    "size_bytes": {"type": "integer"}
                }
            },
            "ProcessRequest": {
                "type": "object",
                "required": ["job_id", "target_ral_code"],
                "properties": {
                    "job_id": {"type": "string", "example": "20251106_120000_a1b2c3d4"},
                    "target_ral_code": {"type": "string", "example": "RAL 3000"},
                    "downsample": {"type": "boolean", "default": False}
                }
            },
            "ProcessResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result_job_id": {"type": "string"},
                    "ral_info": {"type": "object"},
                    "qc_report": {
                        "type": "object",
                        "properties": {
                            "delta_e_statistics": {"type": "object"},
                            "acceptance_rate": {"type": "number"},
                            "passed": {"type": "boolean"}
                        }
                    },
                    "downloads": {"type": "object"}
                }
            },
            "AsyncTaskResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "task_id": {"type": "string", "example": "abc123-def456"},
                    "status_url": {"type": "string", "example": "/api/task/abc123-def456"},
                    "message": {"type": "string", "example": "Task accepted for processing"}
                }
            },
            "TaskStatusResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "task_id": {"type": "string"},
                    "state": {
                        "type": "string",
                        "enum": ["PENDING", "STARTED", "SUCCESS", "FAILURE"],
                        "example": "SUCCESS"
                    },
                    "progress": {"type": "integer", "minimum": 0, "maximum": 100},
                    "result": {"type": "object"},
                    "error": {"type": "string"}
                }
            },
            "DeltaERequest": {
                "type": "object",
                "required": ["color1_rgb", "color2_rgb"],
                "properties": {
                    "color1_rgb": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "example": [255, 0, 0]
                    },
                    "color2_rgb": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "example": [200, 0, 0]
                    }
                }
            },
            "DeltaEResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "delta_e": {"type": "number", "example": 11.83},
                    "interpretation": {"type": "string"},
                    "color1_lab": {"type": "array"},
                    "color2_lab": {"type": "array"}
                }
            },
            "HealthResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "healthy"},
                    "timestamp": {"type": "string"},
                    "palette_loaded": {"type": "boolean"},
                    "colors_count": {"type": "integer"},
                    "celery_connected": {"type": "boolean"}
                }
            },
            "Error": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "example": False},
                    "error": {"type": "string", "example": "Validation error message"},
                    "error_type": {
                        "type": "string",
                        "enum": ["validation_error", "not_found", "file_too_large", "rate_limit", "internal_error"]
                    }
                }
            }
        }
    }
}
