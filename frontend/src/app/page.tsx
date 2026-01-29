"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, BookOpen, Quote, Sparkles, ChevronRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface Source {
  page: string | number;
  source: string;
  content: string;
}

interface Message {
  role: 'user' | 'agent';
  content: string;
  thoughts?: string[];
  sources?: Source[];
}

export default function ChatbotPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'agent',
      content: "Hello! I'm your SupportVector Training Coach. I'm trained on your course materials and will provide verified, hallucination-free answers. What would you like to know about the LLM course?",
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.ok) throw new Error('Failed to connect to AI server');

      const data = await response.json();

      setMessages(prev => [...prev, {
        role: 'agent',
        content: data.answer,
        thoughts: data.thoughts,
        sources: data.sources
      }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, {
        role: 'agent',
        content: "Sorry, I encountered an error connecting to my brain. Please ensure the backend server is running."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#0f1115] text-white">
      {/* Header */}
      <header className="border-b border-white/5 bg-[#1a1d23]/50 backdrop-blur-xl px-6 py-4 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-lg shadow-primary/20">
            <Bot size={22} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">SupportVector Coach</h1>
            <div className="flex items-center gap-1.5 text-[10px] text-green-400 font-medium uppercase tracking-wider">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              Agentic RAG Active
            </div>
          </div>
        </div>
        <div className="flex gap-4">
          <div className="text-right hidden sm:block">
            <p className="text-xs text-muted-foreground">Course Material</p>
            <p className="text-xs font-semibold">Large Language Models</p>
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto px-4 py-8 space-y-8" ref={scrollRef}>
        <div className="max-w-3xl mx-auto space-y-8">
          {messages.map((message, idx) => (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              key={idx}
              className={cn(
                "flex gap-4",
                message.role === 'user' ? "flex-row-reverse" : "flex-row"
              )}
            >
              <div className={cn(
                "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-1",
                message.role === 'user' ? "bg-white/10" : "bg-primary/20 text-primary"
              )}>
                {message.role === 'user' ? <User size={16} /> : <Sparkles size={16} />}
              </div>

              <div className={cn("flex flex-col gap-2 max-w-[85%]", message.role === 'user' ? "items-end" : "items-start")}>
                {/* User Content */}
                {message.role === 'user' && (
                  <div className="chat-bubble-user">
                    {message.content}
                  </div>
                )}

                {/* Agent Content */}
                {message.role === 'agent' && (
                  <div className="space-y-4 w-full">
                    {/* Thoughts (If any) */}
                    {message.thoughts && message.thoughts.length > 0 && (
                      <div className="bg-[#1a1d23] rounded-xl border border-white/5 p-4 overflow-hidden">
                        <button className="flex items-center gap-2 text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3 opacity-60">
                          <div className="w-1 h-1 rounded-full bg-primary" />
                          Agent Reasoning Trace
                        </button>
                        <div className="space-y-2">
                          {message.thoughts.map((thought, tidx) => (
                            <div key={tidx} className="flex gap-2 text-xs text-muted-foreground leading-relaxed animate-in slide-in-from-left duration-300">
                              <ChevronRight size={12} className="shrink-0 mt-0.5 text-primary/40" />
                              {thought}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Main Answer */}
                    <div className="bg-[#1a1d23] border border-white/5 rounded-2xl p-6 text-sm leading-relaxed shadow-xl">
                      <div className="prose prose-invert max-w-none whitespace-pre-wrap">
                        {message.content}
                      </div>
                    </div>

                    {/* Sources */}
                    {message.sources && message.sources.length > 0 && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
                        {message.sources.map((source, sidx) => (
                          <div key={sidx} className="bg-white/5 border border-white/5 rounded-xl p-3 hover:bg-white/10 transition-colors cursor-default group">
                            <div className="flex items-center gap-2 mb-2">
                              <BookOpen size={12} className="text-primary" />
                              <span className="text-[10px] font-bold text-muted-foreground">
                                {source.source}
                              </span>
                              <span className="ml-auto text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded">
                                P{source.page}
                              </span>
                            </div>
                            <p className="text-[10px] text-muted-foreground leading-snug line-clamp-2">
                              {source.content}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          ))}

          {isLoading && (
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center animate-pulse">
                <Bot size={16} className="text-primary" />
              </div>
              <div className="bg-[#1a1d23] rounded-2xl p-4 flex items-center gap-3">
                <Loader2 className="animate-spin text-primary" size={16} />
                <span className="text-xs text-muted-foreground animate-pulse font-medium">Agent is thinking and verifying facts...</span>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Input Area */}
      <footer className="p-6 bg-[#0f1115]">
        <div className="max-w-3xl mx-auto">
          <form onSubmit={handleSubmit} className="relative group">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a technical question about the LLM course..."
              className="w-full bg-[#1a1d23] border border-white/5 rounded-2xl py-4 pl-6 pr-16 text-sm focus:outline-none focus:border-primary/40 focus:ring-4 focus:ring-primary/10 transition-all placeholder:text-muted-foreground/50 shadow-2xl"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="absolute right-2 top-2 p-2.5 bg-primary text-white rounded-xl hover:bg-accent disabled:opacity-30 disabled:hover:bg-primary transition-all shadow-lg shadow-primary/20"
            >
              <Send size={18} />
            </button>
          </form>
          <p className="text-[10px] text-center text-muted-foreground mt-4 font-medium uppercase tracking-widest opacity-40">
            Powered by Gemini 2.0 & LangGraph • Zero Hallucination Mode Active
          </p>
        </div>
      </footer>
    </div>
  );
}
