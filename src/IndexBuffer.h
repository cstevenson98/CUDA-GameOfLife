#pragma once

class IndexBuffer
{
private:
	// ID allocated on VRAM
	unsigned int m_RendererID;
	unsigned int m_Count;
public:
	// Might need to make unsigned long long
	IndexBuffer(const unsigned int* data, unsigned int count);
	~IndexBuffer();

	void Bind() const;
	void Unbind() const;

	inline unsigned int GetCount() const { return m_Count; }
};
