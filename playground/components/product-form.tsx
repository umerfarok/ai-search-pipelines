import type React from "react"
import { useState } from "react"
import { Trash2, ChevronDown, ChevronUp } from "lucide-react"

interface ProductFormProps {
  product: any
  index: number
  fields: { required: any[]; optional: any[] }
  onRemove: (index: number) => void
  onChange: (index: number, field: string, value: string) => void
}

const ProductForm: React.FC<ProductFormProps> = ({ product, index, fields, onRemove, onChange }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [showAdditional, setShowAdditional] = useState(false)

  return (
    <div className="border border-gray-200 rounded-md">
      <div className="flex justify-between items-center p-4 cursor-pointer" onClick={() => setIsOpen(!isOpen)}>
        <span className="font-medium">Product {index + 1}</span>
        <div className="flex items-center">
          <button
            onClick={(e) => {
              e.stopPropagation()
              onRemove(index)
            }}
            className="mr-2 text-red-500 hover:text-red-700"
          >
            <Trash2 className="h-4 w-4" />
          </button>
          {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </div>
      </div>
      {isOpen && (
        <div className="p-4 border-t border-gray-200">
          <div className="space-y-4">
            {fields.required.map((field) => (
              <div key={field.key}>
                <label htmlFor={`${field.key}-${index}`} className="block text-sm font-medium text-gray-700 mb-1">
                  {field.label} {field.required && "*"}
                </label>
                <input
                  id={`${field.key}-${index}`}
                  type="text"
                  value={product[field.key] || ""}
                  onChange={(e) => onChange(index, field.key, e.target.value)}
                  required={field.required}
                  className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            ))}

            <div>
              <button
                type="button"
                onClick={() => setShowAdditional(!showAdditional)}
                className="text-sm text-blue-600 hover:text-blue-800 focus:outline-none"
              >
                {showAdditional ? "Hide" : "Show"} Additional Fields
              </button>
              {showAdditional && (
                <div className="mt-4 space-y-4">
                  {fields.optional.map((field) => (
                    <div key={field.key}>
                      <label htmlFor={`${field.key}-${index}`} className="block text-sm font-medium text-gray-700 mb-1">
                        {field.label}
                      </label>
                      <input
                        id={`${field.key}-${index}`}
                        type="text"
                        value={product[field.key] || ""}
                        onChange={(e) => onChange(index, field.key, e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ProductForm

